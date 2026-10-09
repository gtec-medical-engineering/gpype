"""Season 4, Episode 10: a scikit-learn classifier, fit offline, deployed.

Writes its own two-class recording, a plain HDF5 file, so it needs no
hardware and no download. gp.Epochs cuts one condition per batch run,
and the two are joined in numpy between runs, because a batch pipeline
merges no inputs. That is the shape of training data fit_offline()
exists for. The fitted gp.Sklearn is then deployed through step(), the
call a realtime pipeline makes once per epoch, on trials it never saw.
Requires scikit-learn: pip install scikit-learn
"""

import os
import tempfile

import h5py
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

import gpype as gp
from gpype.common.constants import Constants

#: The recording this example writes and then reads back.
RECORDING = os.path.join(tempfile.gettempdir(), "s4e10.h5")

RATE = 250.0
SIGNAL_CHANNELS = 4
EPOCH_SECONDS = 0.4
EPOCH_SAMPLES = round(EPOCH_SECONDS * RATE)
#: Seconds between one trial's onset and the next, comfortably more
#: than EPOCH_SECONDS, so trials never overlap.
TRIAL_SPACING_SECONDS = 1.0
TRIALS_PER_CLASS = 20
#: Each class's per-channel mean shift during its own epoch window,
#: well separated against unit noise, so there is something to learn.
PATTERN = np.array([3.0, -2.0, 1.5, -1.0])


def write_recording(path: str = RECORDING, seed: int = 0) -> None:
    """Write a two-class synthetic recording as a plain HDF5 file.

    The signal channels carry Gaussian noise, shifted by +PATTERN during
    a "1"-trial's window and by -PATTERN during a "2"-trial's,
    alternating. The last channel is a trigger: zero except a
    single-sample spike of 1.0 or 2.0 at each trial's onset, which
    HDF5Reader turns back into markers by role, with no markers key
    needed in the file at all.

    Args:
        path: File to write.
        seed: For the noise generator, so the recording, and the
            accuracy this example reports, are reproducible.
    """
    rng = np.random.default_rng(seed)
    spacing = round(TRIAL_SPACING_SECONDS * RATE)
    n_trials = TRIALS_PER_CLASS * 2
    total_samples = n_trials * spacing + spacing

    signal = rng.normal(
        scale=1.0, size=(total_samples, SIGNAL_CHANNELS)
    ).astype(np.float64)
    trigger = np.zeros((total_samples, 1))

    for trial in range(n_trials):
        onset = (trial + 1) * spacing - spacing // 2
        label = "1" if trial % 2 == 0 else "2"
        sign = 1.0 if label == "1" else -1.0
        window = slice(onset, onset + EPOCH_SAMPLES)
        signal[window, :] += sign * PATTERN
        trigger[onset, 0] = float(label)

    data = np.column_stack([signal, trigger])
    roles = ["signal"] * SIGNAL_CHANNELS + ["trigger"]
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("data", data=data.T)
        dataset.attrs["sampling_rate"] = RATE
        dataset.attrs["channel_roles"] = np.array(roles, dtype=object)


def cut_condition(path: str, condition: str) -> gp.Result:
    """One condition's trials, as a (time, channel, trial) Result.

    Args:
        path: The recording to read.
        condition: The trigger label to cut epochs around.

    Returns:
        The Collector's Result: SIGNAL_CHANNELS channels, one trial per
        occurrence of *condition*.
    """
    with gp.Pipeline() as p:
        reader = gp.HDF5Reader(
            file_name=path, mode="batch", has_time_row=False
        )
        # The trigger channel already did its job, HDF5Reader derived
        # markers from it at load time, and must not also reach the
        # classifier as a feature: it would encode the label directly
        signal_only = gp.ChannelSelector(roles=Constants.ChannelRoles.SIGNAL)
        # 0.1 s before the marker to 0.3 s after: the window a live
        # Trigger(time_pre=0.1, time_post=0.3) cuts, sample for sample
        epochs = gp.Epochs(condition=condition,
                           tmin=-0.1, tmax=EPOCH_SECONDS - 0.1)
        collector = gp.Collector()
        p.connect(reader, signal_only)
        p.connect(signal_only, epochs)
        p.connect(epochs, collector)
        return p.run()


if __name__ == "__main__":

    write_recording()

    # 1. Cut each condition separately and merge the two in numpy: the
    #    merge happens between runs, which is why fit_offline() takes a
    #    block rather than reading from a pipeline
    result_1 = cut_condition(RECORDING, "1")
    result_2 = cut_condition(RECORDING, "2")
    block = np.concatenate([result_1.data, result_2.data], axis=2)
    trials = [tuple(t) for t in result_1.trials] + [
        tuple(t) for t in result_2.trials
    ]
    print(
        f"Cut {block.shape[2]} trials of {block.shape[0]} samples, "
        f"{block.shape[1]} channels each."
    )

    # 2. A plain sklearn sanity check first: is this separable at all?
    x = block.transpose(2, 0, 1).reshape(block.shape[2], -1)
    y = [label for _, label in trials]
    # 36 trials against 400 features: shrinkage is what makes a linear
    # discriminant well-posed with more features than trials
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    scores = cross_val_score(lda, x, y, cv=5)
    print(
        f"Plain sklearn 5-fold accuracy: {scores.mean():.2f} "
        f"(chance is 0.50)"
    )

    # 3. gp.Sklearn, fit offline, holding out the last two trials of
    #    each class first: the deployed accuracy below has to be
    #    measured on trials the estimator never saw
    holdout = 2
    train_block = np.concatenate(
        [
            result_1.data[:, :, :-holdout],
            result_2.data[:, :, :-holdout],
        ],
        axis=2,
    )
    train_trials = [tuple(t) for t in result_1.trials[:-holdout]] + [
        tuple(t) for t in result_2.trials[:-holdout]
    ]
    test_epochs = [
        (result_1.data[:, :, trial], "1") for trial in range(-holdout, 0)
    ] + [(result_2.data[:, :, trial], "2") for trial in range(-holdout, 0)]

    clf = gp.Sklearn(
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        name="clf",
    )
    clf.fit_offline(
        train_block,
        {
            Constants.Keys.TRIALS: train_trials,
            Constants.Keys.SAMPLING_RATE: RATE,
        },
    )
    print(
        f"gp.Sklearn fitted on {train_block.shape[2]} trials, held out "
        f"{len(test_epochs)} for deployment."
    )

    # 4. Deploy: one prediction per epoch, through step(), the method a
    #    realtime pipeline calls downstream of a Trigger, one epoch at
    #    a time
    classes = clf.state["classes_"]
    correct = 0
    for frame, true_label in test_epochs:
        prediction = clf.step({Constants.Defaults.PORT_IN: frame})
        index = int(prediction[Constants.Defaults.PORT_OUT][0, 0])
        predicted_label = classes[index]
        correct += predicted_label == true_label
        print(f"  true {true_label!r}, predicted {predicted_label!r}")
    print(
        f"Deployed accuracy on held-out trials: {correct}/{len(test_epochs)}"
    )
