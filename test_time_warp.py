from air_hockey import *
import pytest

def test_event_message():
    with pytest.raises(ValueError):
        EventMessage("me", VirtualTime(0),
                     "it", VirtualTime(0),
                     1, {})

    msg = EventMessage("me", VirtualTime(0),
                       "it", VirtualTime(1),
                       1, {})
    assert msg is not None

    msg = EventMessage(sender="me",
                       sendtime=VirtualTime(0),
                       receiver="it",
                       receivetime=(1),
                       sign=True,
                       body={})
    assert msg is not None


def test_test_itself():
    assert True