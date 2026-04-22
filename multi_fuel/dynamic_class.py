"""Dynamically add methods to a class instance from a list of (name, inputs) tuples."""


class CaseDispatcher:
    def __init__(self, method_specs: list[tuple[str, dict]]):
        for method_name, input_dict in method_specs:
            self._attach_method(method_name, dict(input_dict))

    def _attach_method(self, method_name: str, input_dict: dict) -> None:
        def _bound_method(*args, **kwargs):
            print(f'Calling {method_name}: {input_dict}')
            return

        _bound_method.__name__ = method_name
        _bound_method.__qualname__ = f'{type(self).__name__}.{method_name}'
        setattr(self, method_name, _bound_method)


if __name__ == '__main__':
    dispatcher = CaseDispatcher(
        [
            ('a320neo', {'span': 70.0, 'length': 30}),
            ('n3cc', {'span': 80.0, 'length': 10}),
            ('b777e', {'span': 100.0, 'length': 15}),
        ]
    )

    dispatcher.a320neo()
    dispatcher.n3cc()
    dispatcher.b777e()
