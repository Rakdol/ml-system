import enum


class PLATFORM_ENUM(enum.Enum):
    DOCKER = "docker"
    DOCKER_COMPOSE = "docker_compose"
    KUBERNETES = "kubernetes"
    TEST = "test"

    @staticmethod
    def has_value(item):
        return item in [v.value for v in PLATFORM_ENUM.__members__.values()]


def constant(f):
    def fset(self, value):
        raise TypeError

    def fget(self):
        return f()


class _Constants(object):
    pass


CONSTANTS = _Constants()
