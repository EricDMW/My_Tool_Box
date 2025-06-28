# for directly use the functions in the env: to extend the env
from .kos_env import KuramotoOscillatorEnv, KuramotoOscillatorEnvTorch
from .linemsg_env import LineMsgEnv
from .pistonball_env import PistonballEnv
from .wireless_comm_env import WirelessCommEnv
from .ajlatt_env import ajlatt_env

# use the class
from . import kos_env
from . import linemsg_env
from . import pistonball_env
from . import wireless_comm_env
from . import ajlatt_env