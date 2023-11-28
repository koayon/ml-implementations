import random
import time
from typing import Any, Iterable, Optional

State = tuple[int, ...]
Action = tuple[int, int]


class Nim:
    def __init__(self, initial: list[int] = [1, 3, 5, 7]):
        """
        Initialize game board.
        Each game board has
            - `piles`: a list of how many elements remain in each pile
            - `player`: 0 or 1 to indicate which player's turn
            - `winner`: None, 0, or 1 to indicate who the winner is
        """
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles: Iterable[int]) -> list[Action]:
        """
        Nim.available_actions(piles) takes a `piles` list as input
        and returns all of the available actions `(i, j)` in that state.

        Action `(i, j)` represents the action of removing `j` items
        from pile `i` (where piles are 0-indexed).
        """
        actions = []
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.append((i, j))
        return actions

    @classmethod
    def other_player(cls, player: int):
        """
        Nim.other_player(player) returns the player that is not
        `player`. Assumes `player` is either 0 or 1.
        """
        return 0 if player == 1 else 1

    def switch_player(self):
        """
        Switch the current player to the other player.
        """
        self.player = Nim.other_player(self.player)

    def move(self, action: tuple[int, int]):
        """
        Make the move `action` for the current player.
        `action` must be a tuple `(i, j)`.
        """
        pile, count = action

        # Check for errors
        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        # Update pile
        self.piles[pile] -= count
        self.switch_player()

        # Check for a winner
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI:
    def __init__(self, alpha: float = 0.5, epsilon: float = 0.1, gamma: float = 0.99):
        """
        Initialize AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon rate.

        The Q-learning dictionary maps `(state, action)`
        pairs to a Q-value (a number).
         - `state` is a tuple of remaining piles, e.g. (1, 1, 4, 4)
         - `action` is a tuple `(i, j)` for an action
        """
        self.q: dict[tuple[State, Action], float] = dict()
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon
        self.gamma = gamma  # discount rate

    def update(
        self, old_state: list[int], action: Action, new_state: list[int], reward: float
    ):
        """
        Update Q-learning model, given an old state, an action taken
        in that state, a new resulting state, and the reward received
        from taking that action.
        """
        old_state_tuple = tuple(old_state)
        new_state_tuple = tuple(new_state)
        old_q_val = self.get_q_value(old_state_tuple, action)
        best_future_reward_val = self.best_future_reward(new_state_tuple)
        self.update_q_value(
            old_state_tuple, action, old_q_val, reward, best_future_reward_val
        )

    def get_q_value(self, state: State, action: Action) -> float:
        """
        Return the Q-value for the state `state` and the action `action`.
        If no Q-value exists yet in `self.q`, return 0.
        """
        if (state, action) in self.q:
            return self.q[(state, action)]
        else:
            return 0

    def update_q_value(
        self,
        state: State,
        action: Action,
        old_q_val: float,
        reward: float,
        future_rewards: float,
    ) -> None:
        """
        Update the Q-value for the state `state` and the action `action`
        given the previous Q-value `old_q`, a current reward `reward`,
        and an estimate of future rewards `future_rewards`.

        Use the formula:

        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        where `old value estimate` is the previous Q-value,
        `alpha` is the learning rate, and `new value estimate`
        is the sum of the current reward and estimated future rewards.
        """
        new_value_estimate = reward + self.gamma * future_rewards
        modified_value_estimate = old_q_val + self.alpha * (
            new_value_estimate - old_q_val
        )
        self.q[(state, action)] = modified_value_estimate

    def best_future_reward(self, state: State) -> float:
        """
        Given a state `state`, consider all possible `(state, action)`
        pairs available in that state and return the maximum of all
        of their Q-values.

        Use 0 as the Q-value if a `(state, action)` pair has no
        Q-value in `self.q`. If there are no available actions in
        `state`, return 0.
        """
        available_actions = Nim.available_actions(state)
        current_best_future_reward = 0
        for action in available_actions:
            if (state, action) in self.q:
                current_best_future_reward = max(
                    current_best_future_reward, self.q[state, action]
                )

        return current_best_future_reward

    def choose_action(self, state: list[int], epsilon: bool = True) -> Action:
        """
        Given a state `state`, return an action `(i, j)` to take.

        If `epsilon` is `False`, then return the best action
        available in the state (the one with the highest Q-value,
        using 0 for pairs that have no Q-values).

        If `epsilon` is `True`, then with probability
        `self.epsilon` choose a random available action,
        otherwise choose the best action available.

        If multiple actions have the same Q-value, any of those
        options is an acceptable return value.
        """
        available_actions = list(Nim.available_actions(state))
        print(available_actions)

        if not available_actions:
            raise ValueError("This state has no possible actions")

        if epsilon:
            explore = random.random() < self.epsilon
        else:
            explore = False

        if explore:
            next_action = random.choice(available_actions)
            return next_action

        action_scores: dict[Action, float] = {}

        for action in available_actions:
            state_tuple = tuple(state)
            q_val = self.get_q_value(state_tuple, action)
            action_scores[action] = q_val

        # Sort action_scores by values descending
        sorted_action_scores = sorted(
            action_scores.items(), key=lambda item: item[1], reverse=True
        )

        current_next_action = sorted_action_scores[0][0]

        return current_next_action


def train(n: int):
    """
    Train an AI by playing `n` games against itself.
    """

    player = NimAI()

    # Play n games
    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Nim()

        # Keep track of last move made by either player
        last: dict[int, dict[str, Any]] = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None},
        }

        # Game loop
        while True:
            # Keep track of current state and action
            state = game.piles.copy()
            action = player.choose_action(game.piles)

            print("state", state)
            print("action", action)

            # Keep track of last state and action
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            # Make move
            game.move(action)
            new_state = game.piles.copy()

            # When game is over, update Q values with rewards
            if game.winner is not None:
                player.update(state, action, new_state, -1)
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    1,
                )
                break

            # If game is continuing, no rewards yet
            elif last[game.player]["state"] is not None:
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    0,
                )

    print("Done training")

    # Return the trained AI
    return player


def play(ai, human_player: Optional[int] = None):
    """
    Play human game against the AI.
    `human_player` can be set to 0 or 1 to specify whether
    human player moves first or second.
    """

    # If no player order set, choose human's order randomly
    if human_player is None or human_player not in [0, 1]:
        human_player = random.randint(0, 1)

    # Create new game
    game = Nim()

    # Game loop
    while True:
        # Print contents of piles
        print()
        print("Piles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")
        print()

        # Compute available actions
        available_actions = Nim.available_actions(game.piles)
        time.sleep(1)

        # Let human make a move
        if game.player == human_player:
            print("Your Turn")
            while True:
                pile = int(input("Choose Pile: "))
                count = int(input("Choose Count: "))
                if (pile, count) in available_actions:
                    break
                print("Invalid move, try again.")

        # Have AI make a move
        else:
            print("AI's Turn")
            pile, count = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose to take {count} from pile {pile}.")

        # Make move
        game.move((pile, count))

        # Check for winner
        if game.winner is not None:
            print()
            print("GAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return
