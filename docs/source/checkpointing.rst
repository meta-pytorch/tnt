Checkpointing
================================

TorchTNT offers checkpointing via :class:`~torchtnt.framework.callbacks.DistributedCheckpointSaver` which uses `DCP <https://github.com/pytorch/pytorch/tree/main/torch/distributed/checkpoint>`_ under the hood.

.. code-block:: python

    module = nn.Linear(input_dim, 1)
    unit = MyUnit(module=module)
    dcp = DistributedCheckpointSaver(
        dirpath=your_dirpath_here,
        save_every_n_train_steps=100,
        save_every_n_epochs=2,
    )
    # loads latest checkpoint, if it exists
    if latest_checkpoint_dir:
        dcp.restore_from_latest(your_dirpath_here, unit, train_dataloader=dataloader)
    train(
        unit,
        dataloader,
        callbacks=[dcp]
    )

There is built-in support for saving and loading distributed models (DDP, FSDP).

Fully Sharded Data Parallel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The state dict type to be used for checkpointing FSDP modules can be specified in the :class:`~torchtnt.utils.prepare_module.FSDPStrategy`'s state_dict_type argument like so:

.. code-block:: python

    module = nn.Linear(input_dim, 1)
    fsdp_strategy = FSDPStrategy(
        # sets state dict type of FSDP module
        state_dict_type=STATE_DICT_TYPE.SHARDED_STATE_DICT
    )
    module = prepare_fsdp(module, strategy=fsdp_strategy)
    unit = MyUnit(module=module)
    dcp = DistributedCheckpointSaver(
        dirpath=your_dirpath_here,
        save_every_n_epochs=2,
    )
    train(
        unit,
        dataloader,
        # checkpointer callback will use state dict type specified in FSDPStrategy
        callbacks=[dcp]
    )

Or you can manually set this using `FSDP.set_state_dict_type <https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type>`_.

.. code-block:: python

    module = nn.Linear(input_dim, 1)
    module = FSDP(module, ....)
    FSDP.set_state_dict_type(module, StateDictType.SHARDED_STATE_DICT)
    unit = MyUnit(module=module, ...)
    dcp = DistributedCheckpointSaver(
        dirpath=your_dirpath_here,
        save_every_n_epochs=2,
    )
    train(
        unit,
        dataloader,
        callbacks=[dcp]
    )

If you are training only a subset of parameters, and want to save only those subset of weights, you must wrap your fsdp module in a checkpoint wrapper that removes the keys when ``state_dict`` is called. An example wrapper may look like this

.. code-block:: python

    class FSDPCheckpointWrapper(nn.Module):
        """
        Wrapper around FSDP modules that allows us to augment their state_dicts
        prior to checkpointing.
        """

        def __init__(self, fsdp_module, drops: List[str]):
            super().__init__()
            self.fsdp_module = fsdp_module
            self.drops = drops

        def forward(self, *args, **kwargs):
            return self.fsdp_module.forward(*args, **kwargs)

        def __getattr__(self, name: str):
            """
            Forward missing attributes to the wrapped module.
            """

            try:
                # defer to nn.Module's logic
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.fsdp_module, name)

        def __getitem__(self, key: int):
            """
            Forward indexing calls in case the module is an ``nn.Sequential``.
            """

            return self.fsdp_module.__getitem__(key)

        def state_dict(self):
            state_dict = self.fsdp_module.state_dict()

            state_dict_keys = list(state_dict.keys())
            for key in state_dict_keys:
                if not any(drop in key for drop in self.drops):
                    continue
                else:
                    state_dict.pop(key)
            return state_dict

If using the :class:`~torchtnt.framework.auto_unit.AutoUnit`'s strategy to setup your module with FSDP, you will need to wrap this ``FSDPCheckpointWrapper`` on the unit's module after the parent constructor is called.

.. code-block:: python

    class YourAutoUnit(AutoUnit):

        def __init__(self, module, strategy, ...):
            # assuming FSDPStrategy is passed in, this will setup FSDP on the module
            super().__init__(module, strategy, ...)

            # after module has been FSDP wrapped, we wrap FSDPCheckpointWrapper now
            self.module = FSDPCheckpointWrapper(self.module, drop=[...])

This will ensure you can drop keys of unneeded weights. To load this checkpoint, ensure ``strict`` is set to False.

.. code-block:: python

    app_state = ...
    snapshot = Snapshot(path=dirpath)
    snapshot.restore(app_state, strict=False)

Loading State Dict Directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If needing to load the state dict of a module directly, you can do the following:

.. code-block:: python

    snapshot = Snapshot(path=dirpath)
    state_dict = snapshot.get_state_dict_for_key(key="module")  # replace with the attribute name which stores your module. It is `module` for all AutoUnit derived units

Finetuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When finetuning your models, you can pass RestoreOptions to avoid loading optimizers and learning rate schedulers like so:

.. code-block:: python

    dcp = DistributedCheckpointSaver(
        dirpath=your_dirpath_here,
        save_every_n_train_steps=100,
        save_every_n_epochs=2,
    )

    # loads latest checkpoint, if it exists
    if latest_checkpoint_dir:
        dcp.restore_from_latest(
            your_dirpath_here,
            your_unit,
            train_dataloader=dataloader,
            restore_options=RestoreOptions(restore_optimizers=False, restore_lr_schedulers=False)
        )


Best Model by Metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes it may be helpful to keep track of how models perform. This can be done via the BestCheckpointConfig param:

.. code-block:: python

    module = nn.Linear(input_dim, 1)
    unit = MyUnit(module=module)
    dcp = DistributedCheckpointSaver(
        dirpath=your_dirpath_here,
        save_every_n_epochs=1,
        best_checkpoint_config=BestCheckpointConfig(
            monitored_metric="train_loss",
            mode="min"
        )
    )

    train(
        unit,
        dataloader,
        callbacks=[dcp]
    )

By specifying the monitored metric to be "train_loss", the checkpointer will expect the :class:`~torchtnt.framework.unit.TrainUnit` to have a "train_loss" attribute at the time of checkpointing, and it will cast this value to a float and append the value to the checkpoint path name. This attribute is expected to be computed and kept up to date appropriately in the unit by the user.

Later on, the best checkpoint can be loaded via

.. code-block:: python

    DistributedCheckpointSaver.restore_from_best(your_dirpath_here, unit, metric_name="train_loss", mode="min")

If you'd like to monitor a validation metric (say validation loss after each eval epoch during :py:func:`~torchtnt.framework.fit.fit`), you can use the `save_every_n_eval_epochs` flag instead, like so

.. code-block:: python

    dcp = DistributedCheckpointSaver(
        dirpath=your_dirpath_here,
        save_every_n_eval_epochs=1,
        best_checkpoint_config=BestCheckpointConfig(
            monitored_metric="eval_loss",
            mode="min"
        )
    )

And to save only the top three performing models, you can use the existing `keep_last_n_checkpoints` flag like so

.. code-block:: python

    dcp = DistributedCheckpointSaver(
        dirpath=your_dirpath_here,
        save_every_n_eval_epochs=1,
        keep_last_n_checkpoints=3,
        best_checkpoint_config=BestCheckpointConfig(
            monitored_metric="eval_loss",
            mode="min"
        )
    )
