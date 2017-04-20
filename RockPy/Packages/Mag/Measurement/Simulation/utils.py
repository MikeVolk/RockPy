import pandas as pd
import numpy as np

def ThellierStepMaker(steps, tmax=680., ck_every=2, tr_every=2, ac_every=2):
    """
    Creates a pandas DataFrame and a list of measurement steps.

    If filepath is provided a latex document is created

    Magic Lab-treatment codes :
        'TH': 'LT-T-Z'
        'PT': 'LT-T-I'
        'CK': 'LT-PTRM-I'
        'TR': 'LT-PTRM-MD'
        'AC': 'LT-PTRM-Z'
        'NRM': 'LT-NO'
    """

    if isinstance(steps, int):
        steps = np.linspace(20, tmax, steps)

    steps = sorted(list(steps))

    out = pd.DataFrame(columns=('LT-NO', 'LT-T-Z', 'LT-PTRM-I', 'LT-T-I', 'LT-PTRM-Z', 'LT-PTRM-MD'))
    out.loc[0, 'LT-NO'] = 20
    # out.loc[0, 'LT-T-Z'] = 20

    pTH = 20  # previous th step

    for i, t in enumerate(steps[1:]):
        i += 1
        pTH = t

        out.loc[i, 'LT-T-Z'] = t

        if ck_every != 0 and i != 0 and not i % ck_every:
            ck_step = steps[i - ck_every+1]

            try:
                if ck_step == pTH:
                    ck_step = steps[i - ck_every]
                    if i < len(steps):
                        out.loc[i + 1, 'LT-PTRM-I'] = ck_step
                else:
                    out.loc[i, 'LT-PTRM-I'] = ck_step

            except IndexError:
                pass

        out.loc[i, 'LT-T-I'] = t

        if ac_every != 0 and i != 0 and not i % ac_every:
            ac_step = steps[i - ac_every +1]

            if ac_step == pTH:
                ac_step = steps[i - ac_every]
                if i < len(steps):
                    out.loc[i + 1, 'LT-PTRM-Z'] = ac_step
            elif i <= len(steps):
                out.loc[i, 'LT-PTRM-Z'] = ac_step

        if tr_every != 0 and not i % tr_every and not i == 0:
            out.loc[i, 'LT-PTRM-MD'] = t

    return out