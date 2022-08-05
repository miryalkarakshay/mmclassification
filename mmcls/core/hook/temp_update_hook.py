from mmcv.runner import EpochBasedRunner
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.parallel import is_module_wrapper


@HOOKS.register_module()
class TemperatureUpdateHook(Hook):

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        if not isinstance(runner, EpochBasedRunner):
            return
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.backbone.update_temperature()

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
