from __future__ import annotations

from typing import Optional, Sequence, Tuple
from typing import Dict, List

import numpy as np
import pandas as pd
from pymovements.transforms import pos2vel, vnorm


def compute_dispersion(x):
    return np.abs(np.sum(np.max(x, axis=0) - np.min(x, axis=0)))


def compute_fixation_properties(x, v, segmentation):
    onsets, offsets = segmentation_to_events(segmentation)

    v_peaks = np.zeros(onsets.shape) * np.nan
    v_stds = np.zeros(onsets.shape) * np.nan
    dispersions = np.zeros(onsets.shape) * np.nan

    for event_id, (onset, offset) in enumerate(zip(onsets, offsets)):
        if np.isnan(v[onset:offset+1]).all():
            v_peaks[event_id] = np.nan
            v_stds[event_id] = np.nan
        else:
            v_peaks[event_id] = np.max(np.abs(vnorm(v[onset:offset+1], axis=1)))
            v_stds[event_id] = np.nanstd(v[onset:offset+1])

        dispersions[event_id] = compute_dispersion(x[onset:offset+1])

    df = pd.DataFrame(
        data={
            'onset': onsets,
            'offset': offsets,
            'v_std': v_stds,
            'v_peak': v_peaks,
            'dispersion': dispersions,
        },
    )
    return df


def filter_fixations(
    events,
    max_fixation_velocity: float,
    min_fixation_duration: float,
    max_fixation_dispersion: float,
):
    events = events[events.v_peak <= max_fixation_velocity]
    events = events[events.dispersion <= max_fixation_dispersion]

    # FIXME: this works only for a sample rate of 1000 Hz!
    events = events[(events.offset - events.onset + 1) >= min_fixation_duration]

    return events


def filter_saccades(
    events,
    min_saccade_duration: float,
    max_saccade_duration: float,
    min_saccade_peak_velocity: float,
    max_saccade_peak_velocity: float,
):
    events = events[events.v_peak <= max_saccade_peak_velocity]
    events = events[events.v_peak >= min_saccade_peak_velocity]

    # FIXME: this works only for a sample rate of 1000 Hz!
    event_durations = (events.offset - events.onset + 1).values
    events = events[(event_durations >= min_saccade_duration) &
                    (event_durations <= max_saccade_duration)]
    return events


def segmentation_to_events(segmentation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Onsets are the indices of the first True sample of consecutive group.
    onsets = np.where(np.diff(segmentation.astype(int), 1) == 1)[0] + 1
    # Offsets are the indices of the last True sample of consecutive group.
    offsets = np.where(np.diff(segmentation.astype(int), 1) == -1)[0]

    if len(onsets) == 0 and len(offsets) == 0:
        return np.array([]), np.array([])

    # Insert onset at beginning if first offset before first onset.
    if onsets[0] > offsets[0]:
        onsets = np.insert(onsets, 0, 0)

    # Insert offset at end if last onset after last offset.
    if onsets[-1] > offsets[-1]:
        offsets = np.append(offsets, len(segmentation))

    assert (onsets <= offsets).all()
    assert (offsets[:-1] < onsets[1:]).all()

    return onsets, offsets


def events_to_segmentation(events: pd.DataFrame, sequence_length: int) -> np.ndarray:
    segmentation = np.zeros(sequence_length, dtype=bool)

    for _, row in events.iterrows():
        onset, offset = int(row['onset']), int(row['offset'])
        segmentation[onset:offset+1] = True

    return segmentation


def detect_events(
    method: str,
    df: pd.DataFrame,
    is_monocular: bool,
    max_fixation_velocity: float = 20,
    min_fixation_duration: float = 40,
    max_fixation_dispersion: float = 2.7,
    min_saccade_duration: float = 9,
    max_saccade_duration: float = 100,
    min_saccade_peak_velocity: float = 35,
    max_saccade_peak_velocity: float = 1000,
) -> Dict[str, np.ndarray]:

    if method != 'engbert':
        raise NotImplementedError()

    if is_monocular:

        pos_cols = ['x', 'y']
        vel_cols = ['dx', 'dy']
    else:

        pos_cols = ['xr', 'yr']
        vel_cols = ['dxr', 'dyr']

    engbert_saccade_properties, _, _ = microsaccades(
        x=df[pos_cols].values,
        v=df[vel_cols].values,
    )

    engbert_saccade_properties = filter_saccades(
        events=engbert_saccade_properties,
        min_saccade_duration=min_saccade_duration,
        max_saccade_duration=max_saccade_duration,
        min_saccade_peak_velocity=min_saccade_peak_velocity,
        max_saccade_peak_velocity=max_saccade_peak_velocity,
    )

    engbert_saccade_mask = events_to_segmentation(
        events=engbert_saccade_properties,
        sequence_length=len(df),
    )

    ivt_fixation_mask, _ = ivt(
        positions=df[pos_cols].values,
        velocities=df[vel_cols].values,
        threshold=max_fixation_velocity,
    )

    # remove timesteps from fixation mask which are detected as saccades
    ivt_fixation_mask[np.logical_and(engbert_saccade_mask, ivt_fixation_mask)] = False

    ivt_fixation_properties = compute_fixation_properties(
        x=df[pos_cols].values,
        v=df[vel_cols].values,
        segmentation=ivt_fixation_mask,
    )

    ivt_fixation_properties = filter_fixations(
        events=ivt_fixation_properties,
        max_fixation_velocity=max_fixation_velocity,
        min_fixation_duration=min_fixation_duration,
        max_fixation_dispersion=max_fixation_dispersion,
    )

    ivt_fixation_mask = events_to_segmentation(
        sequence_length=len(df),
        events=ivt_fixation_properties,
    )

    event_segmentations = {
        'engbert.saccade': engbert_saccade_mask,
        # 'engbert.fixation': ~issac,
        'ivt.fixation': ivt_fixation_mask,
    }

    # engbert_fixation_properties = compute_fixation_properties(
    #     x=df[['x', 'y']].values,
    #     v=df[['dx', 'dy']].values,
    #     segmentation=event_segmentations['engbert.fixation'],
    # )

    event_properties = {
        'engbert.saccade': engbert_saccade_properties,
        # 'engbert.fixation': engbert_fixation_properties,
        'ivt.fixation': ivt_fixation_properties,
    }

    return event_properties, event_segmentations

    if False:

        sac_right, issac_right, _ = microsaccades(
            x=df[['xr', 'yr']].values,
            v=df[['dxr', 'dyr']].values,
        )

        sac_left, issac_left, _ = microsaccades(
            x=df[['xl', 'yl']].values,
            v=df[['dxl', 'dyl']].values,
        )

        issac_bino = np.logical_and(issac_left, issac_right)
        issac_mono = np.logical_xor(issac_left, issac_right)
        nosac = ~np.logical_or(issac_left, issac_right)

        return {
            'engbert.saccade': issac_bino,
            'engbert.saccade.monocular': issac_mono,
            'engbert.fixation': nosac,
        }


def align_segmentation(
        t_uncut: np.ndarray,
        t_cut: np.ndarray,
        segmentation: np.ndarray,
) -> np.ndarray:
    segmentations_aligned = np.zeros(t_cut.shape, dtype=bool)

    subseq_starts = np.where(np.isin(t_uncut, t_cut[:, 0]))[1]
    subseq_ends = np.where(np.isin(t_uncut, t_cut[:, -1]))[1]

    for subseq_id, (subseq_start, subseq_end) in enumerate(zip(subseq_starts, subseq_ends)):
        segmentations_aligned[subseq_id] = segmentation[subseq_start:subseq_end+1]

    return segmentations_aligned


def align_df(
        t_uncut: np.ndarray,
        t_cut: np.ndarray,
        df: pd.DataFrame,
) -> List[pd.DataFrame]:
    subseq_starts = np.where(np.isin(t_uncut, t_cut[:, 0]))[1]
    subseq_ends = np.where(np.isin(t_uncut, t_cut[:, -1]))[1]

    df_instances = []
    for subseq_id, (subseq_start, subseq_end) in enumerate(zip(subseq_starts, subseq_ends)):
        df_instance = df[(df.offset >= subseq_start) & (df.onset <= subseq_end)].copy()

        df_instance.onset -= subseq_start
        df_instance.offset -= subseq_start

        df_instances.append(df_instance)

    return df_instances


def ivt(
        positions: list[list[float]] | np.ndarray,
        velocities: Optional[list[list[float]] | np.ndarray],
        threshold: float
) -> list[dict]:
    """
    Identification of fixations based on velocity-threshold
    Parameters
    ----------
    positions: array-like
        Continuous 2D position time series.
    velocities: array-like
        Corresponding continuous 2D velocity time series.
    threshold: float
        Velocity threshold.
    Returns
    -------
    fixations: array
        List of fixations
    """
    positions = np.array(positions)
    velocities = np.array(velocities)

    # Make sure positions and velocities have shape (N, 2).
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions need to have shape (N, 2) but shape is {positions.shape}")
    if velocities.ndim != 2 or velocities.shape[1] != 2:
        raise ValueError(f"velocities need to have shape (N, 2) but shape is {velocities.shape}")

    # Check matching shape for positions and velocities.
    if not positions.shape == velocities.shape:
        raise ValueError(f"shape of positions {positions.shape} doesn't match"
                         f"shape of velocities {velocities.shape}")

    # Check if threshold is None.
    if threshold is None:
        raise ValueError("velocity threshold is None")

    # Check if threshold is greater than 0.
    if threshold <= 0:
        raise ValueError("velocity threshold must be greater than 0")

    velocity_norm = vnorm(velocities, axis=1)

    # Mask velocities lower than threshold value as True.
    fixation_mask = velocity_norm < threshold
    onsets, offsets = segmentation_to_events(fixation_mask)

    fixations = []

    for onset, offset in zip(onsets, offsets):
        fixation_points = positions[onset:offset]

        fixation = {
            'onset': onset,
            'offset': offset,
        }

        fixations.append(fixation)

    return fixation_mask, fixations


def microsaccades(
        x: np.ndarray,
        v: Optional[np.ndarray] = None,
        eta: Optional[Sequence[float]] = None,
        lam: float = 6,
        min_duration: int = 6,
        sampling_rate: float = 1000,
        pos2vel_method: Optional[str] = 'smooth',
        sigma_method: Optional[str] = 'engbert2015',
        min_eta: float = 1e-10,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Compute (micro-)saccades from raw samples
    adopted from Engbert et al Microsaccade Toolbox 0.9
    von Hans Trukenbrod empfohlen fuer 1000Hz: lam=6, min_duration=6

    :param x: array of shape (N,2) (x und y screen or visual angle coordinates
        of N samples in *chronological* order)
    :param v: TODO
    :param lam: lambda-factor for relative velocity threshold computation
    :param mindur: minimal saccade duration
    :param sampling_rate: sampling frequency of the eyetracker in Hz
    :param threshold: if None: data-driven velocity threshold; if tuple of
        floats: used to compute elliptic threshold
    :param pos2vel_method: TODO
    :param sigma_method: TODO
    :returns:
        - sac - list of arrays of shape (7,): (1) saccade onset, (2) saccade
            offset, (3) peak velocity, (4) horizontal component (dist from first
            to last sample of the saccade), (5) vertical component,
            (6) horizontal amplitude (dist from leftmost to rightmost sample),
            (7) vertical amplitude
        - issac - array of shape (N,): codes whether a sample of x belongs to
            saccade (1) or not (0)
        - radius - horizontal semi-axis of elliptic threshold; vertical
            semi-axis
    """
    x = np.array(x)

    if v is None:
        v = pos2vel(x, sampling_rate=sampling_rate, method=pos2vel_method)
    else:
        v = np.array(v)
        if x.shape != v.shape:
            raise ValueError('x.shape and v.shape do not match')

    if eta is None:
        eta = compute_sigma(v, method=sigma_method)
    else:
        if len(eta) != 2:
            raise ValueError('threshold needs to be two-dimensional')
        eta = np.array(eta)

    if (eta < min_eta).any():
        raise ValueError(
            f'Threshold eta does not provide enough variance'
            f' ({eta} < {min_eta})')

    # radius of elliptic threshold
    radius = lam * eta

     # test is <1 iff sample within ellipse
    test = np.power((v[:, 0] / radius[0]), 2) + np.power((v[:, 1] / radius[1]), 2)
    # indices of candidate saccades
    # runtime warning because of nans in test
    # => is ok, the nans come from nans in x
    indx = np.where(np.greater(test,1))[0]

    # Initialize saccade variables
    N = len(indx)  # number of saccade candidates
    nsac = 0
    sac = []
    dur = 1
    a = 0  # (potential) saccade onset
    k = 0  # (potential) saccade offset, will be looped over
    issac = np.zeros(len(x), dtype=bool) # codes if row in x is a saccade

    # Loop over saccade candidates
    while k < N - 1:
        # check for ongoing saccade candidate and increase duration by one
        if indx[k + 1] - indx[k] == 1:
            dur += 1

        # else saccade has ended
        else:
            # check minimum duration criterion (exception: last saccade)
            if dur  >= min_duration:
                nsac += 1
                s = {}  # entry for this saccade
                s['onset'] = indx[a]  # saccade onset
                s['offset'] = indx[k]  # saccade offset
                sac.append(s)
                # code as saccade from onset to offset
                issac[indx[a]:indx[k]+1] = 1

            a = k + 1  # potential onset of next saccade
            dur = 1  # reset duration
        k += 1

    # Check minimum duration for last microsaccade
    if dur >= min_duration:
        nsac += 1
        s = {} # entry for this saccade
        s['onset'] = indx[a] # saccade onset
        s['offset'] = indx[k] # saccade offset
        sac.append(s)
        # code as saccade from onset to offset
        issac[indx[a]:indx[k]+1] = 1

    #sac = np.array(sac)

    if nsac > 0:
        # Compute peak velocity, horizontal and vertical components
        for s in range(nsac): # loop over saccades
            # Onset and offset for saccades
            a = int(sac[s]['onset']) # onset of saccade s
            b = int(sac[s]['offset']) # offset of saccade s
            idx = range(a,b+1) # indices of samples belonging to saccade s
            #print(list(idx))
            # Saccade peak velocity (vpeak)
            sac[s]['v_peak'] = np.max(np.sqrt(np.power(v[idx, 0], 2) + np.power(v[idx, 1], 2)))
            # saccade length measured as distance between first (onset) and last (offset) sample
            sac[s]['length_x'] = x[b,0]-x[a,0]
            sac[s]['length_y'] = x[b,1]-x[a,1]
            # Saccade amplitude: saccade length measured as distance between leftmost and rightmost (bzw. highest and lowest) sample
            minx = np.min(x[idx,0]) # smallest x-coordinate during saccade
            maxx = np.max(x[idx,0])
            miny = np.min(x[idx,1])
            maxy = np.max(x[idx,1])
            signx = np.sign(np.where(x[idx,0]==maxx)[0][0] - np.where(x[idx,0]==minx)[0][0]) # direction of saccade; np.where returns tuple; there could be more than one minimum/maximum => chose the first one
            signy = np.sign(np.where(x[idx,1]==maxy)[0][0] - np.where(x[idx,1]==miny)[0][0]) #
            sac[s]['amp_x'] = signx * (maxx-minx) # x-amplitude
            sac[s]['amp_y'] = signy * (maxy-miny) # y-amplitude

    if len(sac) > 0:
        sac = pd.DataFrame.from_records(sac)
    else:
        sac = pd.DataFrame(columns=['onset', 'offset', 'v_peak', 'length_x', 'length_y', 'amp_x', 'amp_y'])

    return sac, issac, radius


def compute_sigma(v: np.ndarray, method='engbert2015'):
    """
    Compute variation in velocity (sigma) by taking median-based std of x-velocity

    engbert2003:
    Ralf Engbert and Reinhold Kliegl: Microsaccades uncover the orientation of
    covert attention

    TODO: add detailed descriptions of all methods
    """
    # TODO: use axis instead of explicit x/y-operations

    if method == 'std':
        thx = np.nanstd(v[:,0])
        thy = np.nanstd(v[:,1])

    elif method == 'mad':
        thx = np.nanmedian(np.absolute(v[:,0] - np.nanmedian(v[:,0])))
        thy = np.nanmedian(np.absolute(v[:,1] - np.nanmedian(v[:,1])))

    elif method == 'engbert2003':
        thx = np.sqrt(np.nanmedian(np.power(v[:,0], 2))
                      - np.power(np.nanmedian(v[:,0]), 2))
        thy = np.sqrt(np.nanmedian(np.power(v[:,1], 2))
                      - np.power(np.nanmedian(v[:,1]), 2))

    elif method == 'engbert2015':
        thx = np.sqrt(np.nanmedian(np.power(v[:,0] - np.nanmedian(v[:,0]), 2)))
        thy = np.sqrt(np.nanmedian(np.power(v[:,1] - np.nanmedian(v[:,1]), 2)))

    else:
        valid_methods = ['std', 'mad', 'engbert2003', 'engbert2015']
        raise ValueError(
            'Method "{method}" not implemented. Valid methods: {valid_methods}')

    return np.array([thx, thy])
