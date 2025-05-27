import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import gym
from gym import spaces


class SimpleMMORPGEnv:
    def __init__(self):
        # สถานะของเกม
        self.player_hp = 100
        self.player_mp = 50
        self.player_level = 1
        self.player_exp = 0
        self.player_gold = 0
        self.player_x = 5
        self.player_y = 5

        # ขนาดแผนที่
        self.map_size = 10

        # มอนสเตอร์
        self.monsters = self._spawn_monsters()

        # Action space: 0=North, 1=South, 2=East, 3=West, 4=Attack, 5=Rest
        self.action_space = spaces.Discrete(6)

        # State space
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(8,), dtype=np.float32
        )

    def _spawn_monsters(self):
        monsters = []
        for _ in range(5):
            x = random.randint(0, self.map_size - 1)
            y = random.randint(0, self.map_size - 1)
            monsters.append({
                'x': x, 'y': y, 'hp': 30, 'level': 1,
                'reward_exp': 10, 'reward_gold': 5
            })
        return monsters

    def get_state(self):
        # สร้างสถานะปัจจุบัน
        nearest_monster = self._find_nearest_monster()
        monster_distance = nearest_monster['distance'] if nearest_monster else 10
        monster_hp = nearest_monster['hp'] if nearest_monster else 0

        return np.array([
            self.player_hp / 100.0,
            self.player_mp / 50.0,
            self.player_level / 10.0,
            self.player_exp / 100.0,
            self.player_x / self.map_size,
            self.player_y / self.map_size,
            monster_distance / 10.0,
            monster_hp / 30.0
        ], dtype=np.float32)

    def _find_nearest_monster(self):
        if not self.monsters:
            return None

        nearest = None
        min_distance = float('inf')

        for monster in self.monsters:
            distance = abs(self.player_x - monster['x']) + abs(self.player_y - monster['y'])
            if distance < min_distance:
                min_distance = distance
                nearest = monster.copy()
                nearest['distance'] = distance

        return nearest

    def step(self, action):
        reward = -0.1  # ค่าใช้จ่ายพื้นฐานต่อการกระทำ
        done = False

        # การเคลื่อนไหว
        if action == 0 and self.player_y < self.map_size - 1:  # North
            self.player_y += 1
        elif action == 1 and self.player_y > 0:  # South
            self.player_y -= 1
        elif action == 2 and self.player_x < self.map_size - 1:  # East
            self.player_x += 1
        elif action == 3 and self.player_x > 0:  # West
            self.player_x -= 1
        elif action == 4:  # Attack
            reward += self._attack_monster()
        elif action == 5:  # Rest
            self.player_hp = min(100, self.player_hp + 10)
            self.player_mp = min(50, self.player_mp + 5)
            reward += 1

        # ตรวจสอบการอัพเลเวล
        if self.player_exp >= 100:
            self.player_level += 1
            self.player_exp = 0
            self.player_hp = 100
            self.player_mp = 50
            reward += 50
            print(f"Level Up! Now level {self.player_level}")

        # เงื่อนไขจบเกม
        if self.player_hp <= 0:
            done = True
            reward -= 100
        elif self.player_level >= 5:
            done = True
            reward += 200

        return self.get_state(), reward, done, {}

    def _attack_monster(self):
        nearest_monster = self._find_nearest_monster()
        if not nearest_monster or nearest_monster['distance'] > 1:
            return -5  # โจมตีพลาด

        # หามอนสเตอร์ที่ใกล้ที่สุด
        for i, monster in enumerate(self.monsters):
            if (monster['x'] == nearest_monster['x'] and
                    monster['y'] == nearest_monster['y']):

                # สุ่มความเสียหาย
                damage = random.randint(15, 25)
                monster['hp'] -= damage

                if monster['hp'] <= 0:
                    # มอนสเตอร์ตาย
                    exp_gained = monster['reward_exp']
                    gold_gained = monster['reward_gold']

                    self.player_exp += exp_gained
                    self.player_gold += gold_gained

                    # เอามอนสเตอร์ออก
                    self.monsters.pop(i)

                    # สร้างมอนสเตอร์ใหม่
                    self._spawn_new_monster()

                    return exp_gained + gold_gained
                else:
                    # มอนสเตอร์โจมตีกลับ
                    counter_damage = random.randint(5, 15)
                    self.player_hp -= counter_damage
                    return 5  # รางวัลการโจมตี

        return -5

    def _spawn_new_monster(self):
        x = random.randint(0, self.map_size - 1)
        y = random.randint(0, self.map_size - 1)
        level = min(5, self.player_level + random.randint(-1, 1))
        self.monsters.append({
            'x': x, 'y': y,
            'hp': 30 + level * 10,
            'level': level,
            'reward_exp': 10 + level * 5,
            'reward_gold': 5 + level * 2
        })

    def reset(self):
        self.player_hp = 100
        self.player_mp = 50
        self.player_level = 1
        self.player_exp = 0
        self.player_gold = 0
        self.player_x = 5
        self.player_y = 5
        self.monsters = self._spawn_monsters()
        return self.get_state()

    def render(self):
        # แสดงสถานะปัจจุบัน
        print(f"Level: {self.player_level}, HP: {self.player_hp}, MP: {self.player_mp}")
        print(f"EXP: {self.player_exp}/100, Gold: {self.player_gold}")
        print(f"Position: ({self.player_x}, {self.player_y})")
        print(f"Monsters nearby: {len(self.monsters)}")
        print("-" * 40)


# Q-Learning Agent
class QLearningAgent:
    def __init__(self, action_size=6, learning_rate=0.1,
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01

        # Q-table สำหรับเก็บค่า Q-values
        self.q_table = defaultdict(lambda: np.zeros(action_size))

    def get_action(self, state):
        # Epsilon-greedy policy
        state_key = self._state_to_key(state)

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state_key])

    def _state_to_key(self, state):
        # แปลง state เป็น key สำหรับ Q-table
        # ปัดเศษเพื่อลดจำนวน state
        return tuple(np.round(state, 1))

    def learn(self, state, action, reward, next_state, done):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        # Q-Learning update rule
        current_q = self.q_table[state_key][action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state_key])

        # Update Q-value
        self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ฟังก์ชันฝึกฝน Q-Learning
def train_q_learning(episodes=1000):
    env = SimpleMMORPGEnv()
    agent = QLearningAgent()

    scores = []
    levels_reached = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 500

        while steps < max_steps:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        scores.append(total_reward)
        levels_reached.append(env.player_level)

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            avg_level = np.mean(levels_reached[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, "
                  f"Average Level: {avg_level:.2f}, Epsilon: {agent.epsilon:.3f}")

    return agent, scores, levels_reached


# ฟังก์ชันทดลองเล่นแบบสุ่ม
def test_random_play():
    print("=== ทดลองเล่นแบบสุ่ม ===")
    env = SimpleMMORPGEnv()
    state = env.reset()

    print("สถานะเริ่มต้น:")
    env.render()

    for step in range(20):
        print(f"\nStep {step + 1}:")
        action = random.randint(0, 5)
        action_names = ['North', 'South', 'East', 'West', 'Attack', 'Rest']
        print(f"Action: {action_names[action]}")

        state, reward, done, _ = env.step(action)
        print(f"Reward: {reward:.2f}")
        env.render()

        if done:
            print("Game Over!")
            break


# ฟังก์ชันแสดงผลกราฟ
def plot_training_results(scores, levels):
    plt.figure(figsize=(15, 5))

    # กราฟ Score
    plt.subplot(1, 3, 1)
    plt.plot(scores, alpha=0.6, color='blue')
    # Moving average
    if len(scores) >= 100:
        moving_avg = np.convolve(scores, np.ones(100) / 100, mode='valid')
        plt.plot(range(99, len(scores)), moving_avg, color='red', linewidth=2, label='Moving Average (100)')
        plt.legend()
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)

    # กราฟ Level
    plt.subplot(1, 3, 2)
    plt.plot(levels, alpha=0.6, color='green')
    if len(levels) >= 100:
        moving_avg = np.convolve(levels, np.ones(100) / 100, mode='valid')
        plt.plot(range(99, len(levels)), moving_avg, color='red', linewidth=2, label='Moving Average (100)')
        plt.legend()
    plt.title('Levels Reached')
    plt.xlabel('Episode')
    plt.ylabel('Level')
    plt.grid(True, alpha=0.3)

    # กราฟแสดง Distribution
    plt.subplot(1, 3, 3)
    plt.hist(levels, bins=range(1, max(levels) + 2), alpha=0.7, color='purple', edgecolor='black')
    plt.title('Level Distribution')
    plt.xlabel('Level')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ฟังก์ชันประเมินผล Agent
def evaluate_agent(agent, episodes=100):
    """ประเมินประสิทธิภาพของ agent"""
    env = SimpleMMORPGEnv()
    scores = []
    levels = []

    # ปิด exploration
    original_epsilon = agent.epsilon
    agent.epsilon = 0

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 500

        while steps < max_steps:
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

            if done:
                break

        scores.append(total_reward)
        levels.append(env.player_level)

    # คืนค่า epsilon
    agent.epsilon = original_epsilon

    print(f"\n=== การประเมิน Agent ({episodes} episodes) ===")
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Average Level: {np.mean(levels):.2f} ± {np.std(levels):.2f}")
    print(f"Max Level Reached: {max(levels)}")
    print(f"Success Rate (Level 3+): {sum(1 for l in levels if l >= 3) / len(levels) * 100:.1f}%")
    print(f"Success Rate (Level 5): {sum(1 for l in levels if l >= 5) / len(levels) * 100:.1f}%")

    return scores, levels


if __name__ == "__main__":
    print("🎮 RL MMORPG Tutorial - บทที่ 1 และ 2 🎮")
    print("=" * 50)

    # 1. ทดลองเล่นแบบสุ่ม
    test_random_play()

    print("\n" + "=" * 50)
    print("เริ่มฝึกสอน Q-Learning Agent...")
    print("=" * 50)

    # 2. ฝึกสอน Q-Learning
    q_agent, q_scores, q_levels = train_q_learning(1000)

    # 3. แสดงผลกราฟ
    print("\nแสดงผลการฝึกสอน...")
    plot_training_results(q_scores, q_levels)

    # 4. ประเมินผล Agent
    print("\nประเมินผล Q-Learning Agent...")
    eval_scores, eval_levels = evaluate_agent(q_agent, 100)

    # 5. ทดลองดู Agent ที่ฝึกแล้วเล่น
    print("\n" + "=" * 50)
    print("ทดลองดู Q-Learning Agent เล่น:")
    print("=" * 50)

    env = SimpleMMORPGEnv()
    state = env.reset()
    agent = q_agent
    agent.epsilon = 0  # ปิด exploration

    print("สถานะเริ่มต้น:")
    env.render()

    for step in range(30):
        action = agent.get_action(state)
        action_names = ['North', 'South', 'East', 'West', 'Attack', 'Rest']
        print(f"\nStep {step + 1}: Action = {action_names[action]}")

        state, reward, done, _ = env.step(action)
        print(f"Reward: {reward:.2f}")
        env.render()

        if done:
            print("🎉 เกมจบแล้ว!")
            break

    print(f"\n📊 สรุปผลการเรียนรู้:")
    print(f"จำนวน Q-states ที่เรียนรู้: {len(q_agent.q_table)}")
    print(f"คะแนนเฉลี่ยใน 100 episodes สุดท้าย: {np.mean(q_scores[-100:]):.2f}")
    print(f"Level เฉลี่ยใน 100 episodes สุดท้าย: {np.mean(q_levels[-100:]):.2f}")