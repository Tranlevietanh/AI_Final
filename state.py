"""State machine: track state and check CAMERA_REQUIREMENTS."""

from typing import Dict, Optional, Tuple

EXPECTED_MAPPING = {
	0: 0, 1: 0, 2: 0,
	3: 1, 4: 2, 5: 6,
	6: 7, 7: 3, 8: 4, 9: 5,
} #mapping giữa object và slot

# State -> slot requirements to advance
CAMERA_REQUIREMENTS = {
	0: {0: 0, 1: 0, 2: 0},          # state 0 -> state 1
	1: {3: 1, 4: 2},               # state 1 -> state 2
	2: {7: 3, 8: 4, 9: 5},         # state 2 -> state 3
	3: {5: 6, 6: 7},               # state 3 -> state 4 (final)
} #mapping giữa state và slot


class StateController:
	"""Simple state machine: track state and check if all slots for current state are OK."""

	def __init__(self, initial_state: int = 0):
		self.state = int(initial_state)

	def get_required(self) -> Dict[int, int]:
		"""Get required slots for current state."""
		return CAMERA_REQUIREMENTS.get(self.state, {})

	def try_advance(self, slot_decisions: Dict[int, bool]) -> Tuple[bool, Optional[int]]:
		"""Check if all required slots are True; if so, advance state and return (True, new_state).
		Otherwise return (False, None)."""
		reqs = self.get_required()

		# Check all required slots
		for slot_id in reqs.keys():
			if not slot_decisions.get(slot_id, False):
				return False, None

		# All OK -> advance state. Chỉ + 1 state, không thể nhảy vọt
		self.state += 1
		return True, self.state

	def is_final(self) -> bool:
		return self.state >= 4

	def reset(self) -> None:
		self.state = 0


def process_from_slot_states(
    controller: StateController,
    slot_states: Dict[int, Optional[bool]]
):
    if controller.is_final():
        return controller.state, False, None

    advanced, new_state = controller.try_advance(slot_states)
    return controller.state, advanced, new_state



__all__ = ["EXPECTED_MAPPING", "CAMERA_REQUIREMENTS", "StateController", "process_from_slot_states"]

	

