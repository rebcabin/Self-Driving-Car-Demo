from air_hockey import *
import pytest


def test_event_message():

    with pytest.raises(ValueError):
        # receivetime strictly greater than sendtime
        EventMessage("me", 42, "it", 42, 1, {})

    msg = EventMessage("me", 42, "it", 43, 1, {})
    assert msg is not None, f"shorthand form is OK."

    msg = EventMessage(sender="me",
                       sendtime=VirtualTime(0),
                       receiver="it",
                       receivetime=VirtualTime(1),
                       sign=True,
                       body={})
    assert msg is not None, f"mixed shorthand is OK."

    msg2 = EventMessage(sender="me",
                        sendtime=VirtualTime(0),
                        receiver="it",
                        receivetime=VirtualTime(1),
                        sign=False,
                        body={})
    assert msg2 is not None

    assert msg == msg2, f"message equality doesn't depend on sign."

    msg3 = EventMessage(sender=ProcessID("me"),
                        sendtime=VirtualTime(100),
                        receiver=ProcessID("it"),
                        receivetime=VirtualTime(150),
                        sign=False,
                        body=Body({}))
    assert msg3 is not None, f"fullform is OK"

    assert msg3 > msg2, f"virtual-time strict comparison is OK."
    assert msg3 >= msg2, f"virtual-time non-strict comparison is OK."

    msg4 = EventMessage(sender=ProcessID("me"),
                        sendtime=VirtualTime(100),
                        receiver=ProcessID("it"),
                        receivetime=VirtualTime(150),
                        sign=False,
                        body=Body({'worcestershire': 'sauce'}))

    assert not msg3 == msg4, f"message equality does depend on body."
    assert msg3 != msg4, f"message inequality operator if OK."


def test_twstate():
    state = State(sender=ProcessID("me"),
                  sendtime=VirtualTime(100),
                  body=Body({'a': 1, 'sauce': 'steak'}))
    assert state is not None

    state2 = State(sender=ProcessID("me"),
                   sendtime=VirtualTime(180),
                   body=Body({'heinz': 57}))

    assert not state2 < state, f"state timestamp lt comparison is OK."
    assert state2 >= state, f"state timestamp ge is OK."
    assert state2 > state, f"state timestamp gt is OK."


def test_twqueue():
    q = TWQueue()
    m = EventMessage("me", 100, "it", 150, True, {'dressing': 'caesar'})
    q.insert(m)
    debug_me = q.vts()
    assert debug_me == [150], f"insert into empty is OK"
    mm = EventMessage("me", 100, "it", 150, False, {'dressing': 'caesar'})
    mm.vt = mm.receive_time
    q.insert(mm)
    assert q.vts() == [], f"annihilation happens."
    assert q.annihilation, f"annihilation flag is set."
    q.insert(m)
    assert q.vts() == [150], f"re-insert into empty is OK"
    q.insert(EventMessage("me", 100, "alice", 150, True, {'dressing': 'mayo'}))
    assert q.vts() == [150]
    assert len(q.elements[150]) == 2



def test_input_queue():
    q = InputQueue()
    m = EventMessage("me", 100, "it", 150, True, {'dressing': 'caesar'})
    q.insert(m)
    assert q.vts() == [150], f"insert into empty is OK"
    mm = EventMessage("me", 100, "it", 150, False, {'dressing': 'caesar'})
    q.insert(mm)
    assert q.vts() == [], f"annihilation happens."
    assert q.annihilation, f"annihilation flag is set."


def test_output_queue():
    pass


def test_sched_queue():
    pass