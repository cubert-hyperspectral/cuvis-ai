# TODO make yaml serializer
import inspect


class Serializable:
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Store the original __init__ signature before it gets wrapped
        cls._original_init_signature = inspect.signature(cls.__init__)

    def __init__(self, *args, **kwargs) -> None:
        sig = getattr(self.__class__, "_original_init_signature", None)
        if sig:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            self.hparams = {}
            for name, value in bound.arguments.items():
                if name == "self":
                    continue
                parameter = sig.parameters.get(name)
                if parameter and parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                    if value:
                        self.hparams[name] = value
                    continue
                if parameter and parameter.kind is inspect.Parameter.VAR_KEYWORD:
                    if value:
                        self.hparams.update(value)
                    continue
                self.hparams[name] = value
        else:
            self.hparams = {}
