import numpy as np

# 地图大小：5 × 5
# . . . . .
# . . . . .
# . . . . .
# . . . . .
# S X X X G


class CliffWorld:
    def __init__(self):
        # start and end point
        self.start = (4, 0)
        self.goal = (4, 4)

        # boundary
        self.rows = 5
        self.cols = 5

        # cliffs
        self.cliffs = {(4, 1), (4, 2), (4, 3)}

        # agent position
        self.agent_pos = self.start

    def reset(self):
        self.agent_pos = self.start

        return self.agent_pos

    def step(self, action):
        # agent position - current
        row, col = self.agent_pos

        # action
        # 0:up, 1:down, 2:left, 3:right
        if action == 0:
            new_pos = (row - 1, col)
        elif action == 1:
            new_pos = (row + 1, col)
        elif action == 2:
            new_pos = (row, col - 1)
        elif action == 3:
            new_pos = (row, col + 1)
        else:
            raise ValueError(f"Input action is: {action}, invalid.")

        # special case
        done = False

        # case 1: Out of boundary
        if (
            new_pos[0] < 0
            or new_pos[0] >= self.rows
            or new_pos[1] < 0
            or new_pos[1] >= self.cols
        ):
            self.agent_pos = (row, col)
            reward = -1

        # case 2: arrival goal
        elif new_pos == self.goal:
            self.agent_pos = new_pos
            reward = -1
            done = True

        # case 3: get into cliff
        elif new_pos in self.cliffs:
            self.agent_pos = self.start
            reward = -100
        else:
            self.agent_pos = new_pos
            reward = -1

        return self.agent_pos, reward, done

    def render(self):
        grid = ""
        for row in range(self.rows):
            for col in range(self.cols):
                pos_tmp = (row, col)
                # agent
                if pos_tmp == self.agent_pos:
                    grid += " A"
                # cliff
                elif pos_tmp in self.cliffs:
                    grid += " X"
                # start
                elif pos_tmp == self.start:
                    grid += " S"
                # goal
                elif pos_tmp == self.goal:
                    grid += " G"
                # passage
                else:
                    grid += " ."

            # a newline
            grid += "\n"

        print(grid)


if __name__ == "__main__":
    print("Start.... \n")
    env = CliffWorld()
    env.agent_pos = (3, 2)
    state, reward, done = env.step(3)
    print(state, reward, done, "\n\n")
    # env.render()

    env.agent_pos = (3, 2)
    state, reward, done = env.step(1)
    print(state, reward, done, "\n\n")
    # env.render()

    env.agent_pos = (3, 4)
    state, reward, done = env.step(1)
    print(state, reward, done, "\n\n")
    # env.render()

    env.agent_pos = (4, 0)
    state, reward, done = env.step(3)
    print(state, reward, done, "\n\n")
    # env.render()

    env.agent_pos = (0, 0)
    state, reward, done = env.step(0)
    print(state, reward, done, "\n\n")
    # env.render()
