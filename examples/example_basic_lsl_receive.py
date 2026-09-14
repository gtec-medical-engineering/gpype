# --------------------------------------------------------------
# Example file example_basic_lsl_receive.py
# Receiving a Lab Streaming Layer stream into a g.Pype pipeline
# --------------------------------------------------------------
#
# The receiving half of the LSL pair. Run example_basic_lsl_send.py
# first to publish a stream, then run this one to read it back.
#
# LSL is how a pipeline gets data from something that is not g.Pype:
# another vendor's amplifier, MATLAB, a stimulus program, or a second
# g.Pype process on another machine. LslReceiver resolves a stream and
# hands its samples to the pipeline like any other source, so everything
# downstream -- filters, scopes, writers -- is unchanged.
#
# Channel names and types come from the stream description, so a g.Pype
# stream read back keeps the names it was published with.
#
# One thing to be clear about: what arrives this way is somebody else's
# data. LSL carries no per-sample authentication, so a stream says what
# it is and cannot prove it.

import gpype as gp

# Resolve by type. Give stream_name= instead when more than one stream
# of this type is on the network and you need a particular one.
STREAM_TYPE = "EEG"

if __name__ == "__main__":

    app = gp.MainApp()
    p = gp.Pipeline()

    # channel_count and sampling_rate are read from the stream when they
    # are not given. Pass them to state what you expect instead.
    source = gp.LslReceiver(stream_type=STREAM_TYPE)

    scope = gp.TimeSeriesScope(amplitude_limit=100, time_window=10)

    p.connect(source, scope)
    app.add_widget(scope)

    p.start()
    app.run()
    p.stop()
    p.close()
