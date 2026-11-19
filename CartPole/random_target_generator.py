import numpy as np
from scipy.interpolate import BPoly, interp1d
import copy
import hashlib


def Generate_Random_Trace_Function(
        length_of_experiment,
        rtf_rng,
        track_relative_complexity,
        interpolation_type,
        turning_points,
        turning_points_period,
        start_random_target_position_at,
        end_random_target_position_at,
        used_track_fraction,
):
    from CartPole.cartpole_parameters import TrackHalfLength
    pista_estatica = float(TrackHalfLength)

    # 1. Clonar el generador aleatorio para independizarlo del bucle de CartPole
    rng_local = copy.deepcopy(rtf_rng)

    # 2. Manejo seguro de turning_points (Copia profunda obligatoria)
    if turning_points is None or (hasattr(turning_points, '__len__') and len(turning_points) == 0):
        valor_estabilizado = round(length_of_experiment * track_relative_complexity, 4)
        number_of_turning_points = int(np.floor(valor_estabilizado))

        y = rng_local.uniform(-1.0, 1.0, number_of_turning_points)
        y = y * used_track_fraction * pista_estatica

        if number_of_turning_points == 0:
            y = np.append(y, 0.0)
            y = np.append(y, 0.0)
        elif number_of_turning_points == 1:
            val = start_random_target_position_at if start_random_target_position_at is not None else end_random_target_position_at
            if val is None: val = 0.0
            y = np.array([val, val])
        else:
            if start_random_target_position_at is not None: y[0] = start_random_target_position_at
            if end_random_target_position_at is not None: y[-1] = end_random_target_position_at
    else:
        # Forzar una copia física en memoria (.copy()) para que nadie afuera pueda mutar 'y'
        y_original = np.atleast_1d(turning_points)
        if len(y_original) == 0:
            raise ValueError('You should not be here!')
        elif len(y_original) == 1:
            y = np.array([y_original[0], y_original[0]])
        else:
            y = y_original.copy()

    random_samples = len(y) - 2 if len(y) - 2 >= 0 else 0

    if turning_points_period == 'random':
        t_init = np.sort(rng_local.uniform(0.0, 1.0, random_samples))
        t_init = np.insert(t_init, 0, 0.0)
        t_init = np.append(t_init, 1.0)
    elif turning_points_period == 'regular':
        t_init = np.linspace(0, 1.0, num=random_samples + 2, endpoint=True)
    else:
        raise NotImplementedError('Unknown turning_points_period variable')

    t_init = t_init * length_of_experiment

    # Asegurar que los vectores internos estén congelados en memoria
    t_init_congelado = t_init.copy()
    y_congelado = y.copy()

    # 3. AUDITORÍA MATEMÁTICA ABSOLUTA
    # Generamos un hash de los datos puros. Si el hash coincide, la curva ES la misma.
    datos_curva = f"{t_init_congelado.tolist()}-{y_congelado.tolist()}"
    checksum_curva = hashlib.md5(datos_curva.encode()).hexdigest()
    print(f"[AUDITORÍA] Checksum interno de la curva: {checksum_curva}")

    if interpolation_type == '0-derivative-smooth':
        yder = [[y_congelado[i], 0] for i in range(len(y_congelado))]
        random_track_f = BPoly.from_derivatives(t_init_congelado, yder, extrapolate='periodic')
    elif interpolation_type == 'linear':
        random_track_f = interp1d(t_init_congelado, y_congelado, kind='linear', fill_value='extrapolate')
    elif interpolation_type == 'previous':
        random_track_f = interp1d(t_init_congelado, y_congelado, kind='previous', fill_value='extrapolate')
    else:
        raise ValueError('Unknown interpolation type.')

    # Clausura protegida contra mutaciones de tiempo externas
    def random_track_f_truncated(time):
        # Asegura que si le pasan un array mutable, trabajemos sobre una copia estática
        time_local = np.copy(time) if isinstance(time, np.ndarray) else time

        target_position = random_track_f(time_local)
        target_position = np.clip(target_position, -used_track_fraction * pista_estatica,
                                  used_track_fraction * pista_estatica)
        return target_position

    return random_track_f_truncated
