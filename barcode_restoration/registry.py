from barcode_restoration.base import RestorationMethod


_METHODS: dict[str, type[RestorationMethod]] = {}


def register_method(cls: type[RestorationMethod]) -> type[RestorationMethod]:
    name = cls.name.lower()

    if name in _METHODS:
        raise ValueError(f"Method already registered: {name}")

    _METHODS[name] = cls
    return cls


def available_methods() -> tuple[str, ...]:
    return tuple(sorted(_METHODS))


def create_method(name: str) -> RestorationMethod:
    key = name.lower()

    if key not in _METHODS:
        available = ", ".join(available_methods()) or "none"
        raise ValueError(
            f"Unknown restoration method '{name}'. "
            f"Available methods: {available}"
        )

    return _METHODS[key]()
