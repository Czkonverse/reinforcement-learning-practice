# 地图大小：5 × 5
# S . . # .
# . # . # .
# . # . . .
# . . # # .
# . . . . G


class GridWorld:
    def __init__(self):
        # start and end point
        self.start = (0, 0)
        self.goal = (4, 4)

        # boundary
        self.rows = 5
        self.cols = 5

        # walls
        self.walls = {(0, 3), (1, 1), (1, 3), (2, 1), (3, 2), (3, 3)}

        # agent position
        self.agent_pos = self.start

    def reset(self):
        # reset the enviroment
        # agent returns to the start point
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
        # case 1: Out of boundary
        out_of_bounds = False
        if (
            new_pos[0] < 0
            or new_pos[0] >= self.rows
            or new_pos[1] < 0
            or new_pos[1] >= self.cols
        ):
            out_of_bounds = True

        # special case 2: Against the wall
        against_wall = False
        if new_pos in self.walls:
            against_wall = True

        # update agent position
        if out_of_bounds or against_wall:
            self.agent_pos = (row, col)
        else:
            self.agent_pos = new_pos

        # reward
        # out of boundary & against the wall : -5
        # normally move : -1
        # arrival : +100
        done = False
        if out_of_bounds or against_wall:
            reward = -5
        else:
            if self.agent_pos == self.goal:
                reward = 100
                done = True
            else:
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
                # wall
                elif pos_tmp in self.walls:
                    grid += " #"
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
    env = GridWorld()
    env.reset()
    state, reward, done = env.step(3)
    print(state, reward, done, "\n\n")
    env.render()

    print("Reset \n")
    env.reset()
    state, reward, done = env.step(0)
    print(state, reward, done, "\n\n")
    env.render()

    print("Reset \n")
    env.reset()
    state, reward, done = env.step(3)
    print(state, reward, done, "\n\n")
    state, reward, done = env.step(1)
    print(state, reward, done, "\n\n")
    env.render()

    print("Reset \n")
    env.reset()
    env.agent_pos = (4, 3)
    state, reward, done = env.step(3)
    print(state, reward, done, "\n\n")
    env.render()
