import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict


class UltraSimpleMMORPGEnv:
    def __init__(self):
        # สถานะของเกม - ลดให้เหลือแค่สิ่งสำคัญ
        self.player_hp = 100
        self.player_level = 1
        self.player_exp = 0
        self.player_x = 2
        self.player_y = 2

        # ขนาดแผนที่เล็ก
        self.map_size = 5

        # มอนสเตอร์ 1 ตัว ที่ตำแหน่งคงที่
        self.monster_x = 1
        self.monster_y = 1
        self.monster_hp = 3  # แค่ 3 HP!
        self.monster_max_hp = 3

        # Action space: 0=North, 1=South, 2=East, 3=West, 4=Attack
        self.action_size = 5

    def get_state(self):
        # State แบบ SUPER SIMPLE - แค่ 4 ค่า
        distance_to_monster = abs(self.player_x - self.monster_x) + abs(self.player_y - self.monster_y)

        state = [
            min(3, self.player_level - 1),  # 0-3 (level 1-4)
            min(3, self.player_exp // 25),  # 0-3 (exp 0-24, 25-49, 50-74, 75+)
            min(4, distance_to_monster),  # 0-4 (distance to monster)
            min(2, self.monster_hp)  # 0-2 (monster hp)
        ]

        return tuple(state)  # Return เป็น tuple ตรงๆ

    def step(self, action):
        reward = 0
        done = False
        info = {}

        # การเคลื่อนไหว
        if action == 0 and self.player_y > 0:  # North
            self.player_y -= 1
            reward += 0.1
        elif action == 1 and self.player_y < self.map_size - 1:  # South
            self.player_y += 1
            reward += 0.1
        elif action == 2 and self.player_x < self.map_size - 1:  # East
            self.player_x += 1
            reward += 0.1
        elif action == 3 and self.player_x > 0:  # West
            self.player_x -= 1
            reward += 0.1
        elif action == 4:  # Attack
            attack_reward = self._attack_monster()
            reward += attack_reward

        # ตรวจสอบการอัพเลเวล - ลดเงื่อนไข
        if self.player_exp >= 50:  # ลดจาก 100 เป็น 50
            self.player_level += 1
            self.player_exp = 0
            reward += 100  # รางวัลใหญ่!
            print(f"🎉 LEVEL UP! Now level {self.player_level}")

            # สร้างมอนสเตอร์ใหม่
            self._spawn_new_monster()

        # เงื่อนไขจบเกม
        if self.player_level >= 3:  # ลดจาก 5 เป็น 3
            done = True
            reward += 200
            print("🏆 YOU WIN!")

        info['level'] = self.player_level
        info['exp'] = self.player_exp

        return self.get_state(), reward, done, info

    def _attack_monster(self):
        # คำนวณระยะห่าง
        distance = abs(self.player_x - self.monster_x) + abs(self.player_y - self.monster_y)

        if distance > 1:
            return -1  # โจมตีพลาด

        if self.monster_hp <= 0:
            return -2  # มอนสเตอร์ตายแล้ว

        # โจมตีสำเร็จ - แดเมจคงที่
        self.monster_hp -= 1

        if self.monster_hp <= 0:
            # มอนสเตอร์ตาย - ได้ EXP เยอะ!
            exp_gained = 30
            self.player_exp += exp_gained
            print(f"💀 Monster defeated! +{exp_gained} EXP")

            # สร้างมอนสเตอร์ใหม่
            self._spawn_new_monster()

            return 50  # รางวัลใหญ่มาก!
        else:
            # โจมตีโดนแต่ยังไม่ตาย
            return 10

    def _spawn_new_monster(self):
        # สร้างมอนสเตอร์ในตำแหน่งใหม่
        positions = [(0, 0), (0, 4), (4, 0), (4, 4), (1, 3), (3, 1)]
        new_pos = random.choice(positions)
        self.monster_x, self.monster_y = new_pos
        self.monster_hp = self.monster_max_hp
        print(f"👹 New monster spawned at ({self.monster_x}, {self.monster_y})")

    def reset(self):
        self.player_hp = 100
        self.player_level = 1
        self.player_exp = 0
        self.player_x = 2
        self.player_y = 2

        # รีเซ็ตมอนสเตอร์
        self.monster_x = 1
        self.monster_y = 1
        self.monster_hp = self.monster_max_hp

        return self.get_state()

    def render(self):
        print(
            f"Player: Level {self.player_level}, EXP {self.player_exp}/50, Position ({self.player_x},{self.player_y})")
        print(f"Monster: HP {self.monster_hp}/{self.monster_max_hp}, Position ({self.monster_x},{self.monster_y})")

        # แสดงแผนที่
        print("\nMap:")
        for y in range(self.map_size):
            row = ""
            for x in range(self.map_size):
                if x == self.player_x and y == self.player_y:
                    row += "P "
                elif x == self.monster_x and y == self.monster_y and self.monster_hp > 0:
                    row += "M "
                else:
                    row += ". "
            print(row)
        print("-" * 30)


class UltraSimpleQLearningAgent:
    def __init__(self):
        self.action_size = 5
        self.learning_rate = 0.3  # เพิ่มให้เรียนรู้เร็วมาก
        self.discount_factor = 0.95
        self.epsilon = 0.8  # เริ่มสำรวจเยอะ
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.1

        # Q-table
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))

        # สถิติ
        self.action_counts = defaultdict(int)

    def get_action(self, state):
        self.action_counts[state] += 1

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])

        # Update Q-value
        self.q_table[state][action] += self.learning_rate * (target_q - current_q)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_ultra_simple(episodes=1000):
    env = UltraSimpleMMORPGEnv()
    agent = UltraSimpleQLearningAgent()

    scores = []
    levels_reached = []
    exp_gained_list = []
    steps_list = []

    print("🎮 เริ่มฝึกสอน Ultra Simple MMORPG")
    print("🎯 เป้าหมาย: เลเวลอัพจาก 1 → 3")
    print("⚔️  วิธี: เดินไปหามอนสเตอร์และโจมตี")
    print("=" * 50)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 50  # จำกัดให้น้อยมาก

        while steps < max_steps:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        scores.append(total_reward)
        levels_reached.append(info['level'])
        exp_gained_list.append(info['exp'] + (info['level'] - 1) * 50)
        steps_list.append(steps)

        # รายงานผลทุก 100 episodes
        if episode % 100 == 0:
            recent_scores = scores[-100:] if len(scores) >= 100 else scores
            recent_levels = levels_reached[-100:] if len(levels_reached) >= 100 else levels_reached
            recent_steps = steps_list[-100:] if len(steps_list) >= 100 else steps_list

            avg_score = np.mean(recent_scores)
            avg_level = np.mean(recent_levels)
            avg_steps = np.mean(recent_steps)
            level_2_rate = sum(1 for l in recent_levels if l >= 2) / len(recent_levels) * 100
            level_3_rate = sum(1 for l in recent_levels if l >= 3) / len(recent_levels) * 100

            print(f"Episode {episode:4d} | Score: {avg_score:6.1f} | Level: {avg_level:.2f} | "
                  f"Steps: {avg_steps:4.1f} | Lv2: {level_2_rate:4.1f}% | Lv3: {level_3_rate:4.1f}% | ε: {agent.epsilon:.3f}")

            # แสดงตัวอย่าง Q-values
            if episode == 500:
                print("\n📊 ตัวอย่าง Q-values:")
                sample_states = list(agent.q_table.keys())[:5]
                for state in sample_states:
                    q_vals = agent.q_table[state]
                    print(f"State {state}: {q_vals}")
                print()

    return agent, scores, levels_reached, steps_list


def evaluate_ultra_simple(agent, episodes=50):
    env = UltraSimpleMMORPGEnv()
    scores = []
    levels = []
    steps_list = []

    # ปิด exploration
    original_epsilon = agent.epsilon
    agent.epsilon = 0

    print(f"\n🧪 Testing Agent (ε=0) for {episodes} episodes...")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 50

        while steps < max_steps:
            action = agent.get_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            if done:
                break

        scores.append(total_reward)
        levels.append(info['level'])
        steps_list.append(steps)

    # คืนค่า epsilon
    agent.epsilon = original_epsilon

    avg_score = np.mean(scores)
    avg_level = np.mean(levels)
    avg_steps = np.mean(steps_list)
    level_2_rate = sum(1 for l in levels if l >= 2) / len(levels) * 100
    level_3_rate = sum(1 for l in levels if l >= 3) / len(levels) * 100

    print(f"\n=== 📊 ผลการทดสอบ Agent ===")
    print(f"🎯 Average Score: {avg_score:.2f}")
    print(f"⭐ Average Level: {avg_level:.2f}")
    print(f"⏱️  Average Steps: {avg_steps:.1f}")
    print(f"📈 Level 2 Rate: {level_2_rate:.1f}%")
    print(f"🏆 Level 3 Rate: {level_3_rate:.1f}%")
    print(f"🧠 Total Q-states: {len(agent.q_table)}")

    # ประเมินความเก่ง
    if level_3_rate >= 80:
        print("🌟 EXCELLENT: Agent เรียนรู้ได้ยอดเยี่ยม!")
    elif level_3_rate >= 50:
        print("🔥 GOOD: Agent เรียนรู้ได้ดี!")
    elif level_3_rate >= 20:
        print("📈 FAIR: Agent เรียนรู้ได้บ้าง")
    elif level_2_rate >= 50:
        print("📚 BASIC: Agent เรียนรู้พื้นฐานได้")
    else:
        print("😅 POOR: Agent ยังเรียนรู้ไม่ได้")

    return scores, levels


def show_agent_play_demo(agent):
    """แสดงตัวอย่างการเล่นของ Agent"""
    env = UltraSimpleMMORPGEnv()
    state = env.reset()
    agent.epsilon = 0  # ปิด exploration

    print(f"\n🎮 Demo: Agent เล่นเกม")
    print("=" * 40)

    env.render()

    for step in range(30):
        action = agent.get_action(state)
        action_names = ['North', 'South', 'East', 'West', 'Attack']

        print(f"\nStep {step + 1}: {action_names[action]}")
        state, reward, done, info = env.step(action)
        print(f"Reward: {reward:.1f}")
        env.render()

        if done:
            print("🎉 Game Completed!")
            break

        if step >= 29:
            print("⏰ Time limit reached")


def plot_ultra_simple_results(scores, levels, steps_list):
    plt.figure(figsize=(16, 8))

    # กราฟ Score
    plt.subplot(2, 4, 1)
    plt.plot(scores, alpha=0.3, color='blue', linewidth=0.5)
    if len(scores) >= 50:
        moving_avg = np.convolve(scores, np.ones(50) / 50, mode='valid')
        plt.plot(range(49, len(scores)), moving_avg, color='red', linewidth=2)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)

    # กราฟ Level
    plt.subplot(2, 4, 2)
    plt.plot(levels, alpha=0.3, color='green', linewidth=0.5)
    if len(levels) >= 50:
        moving_avg = np.convolve(levels, np.ones(50) / 50, mode='valid')
        plt.plot(range(49, len(levels)), moving_avg, color='red', linewidth=2)
    plt.title('Levels Reached')
    plt.xlabel('Episode')
    plt.ylabel('Level')
    plt.grid(True, alpha=0.3)

    # กราฟ Steps
    plt.subplot(2, 4, 3)
    plt.plot(steps_list, alpha=0.3, color='orange', linewidth=0.5)
    if len(steps_list) >= 50:
        moving_avg = np.convolve(steps_list, np.ones(50) / 50, mode='valid')
        plt.plot(range(49, len(steps_list)), moving_avg, color='red', linewidth=2)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True, alpha=0.3)

    # Level Distribution
    plt.subplot(2, 4, 4)
    level_counts = [levels.count(i) for i in range(1, max(levels) + 1)]
    plt.bar(range(1, max(levels) + 1), level_counts, alpha=0.7, color='purple')
    plt.title('Level Distribution')
    plt.xlabel('Level')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)

    # Success Rate Over Time
    plt.subplot(2, 4, 5)
    window = 100
    level_2_rates = []
    level_3_rates = []

    for i in range(len(levels) - window):
        window_levels = levels[i:i + window]
        level_2_rates.append(sum(1 for l in window_levels if l >= 2) / window * 100)
        level_3_rates.append(sum(1 for l in window_levels if l >= 3) / window * 100)

    if level_2_rates:
        plt.plot(level_2_rates, label='Level 2+', linewidth=2)
        plt.plot(level_3_rates, label='Level 3', linewidth=2)
        plt.legend()
    plt.title('Success Rate Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.grid(True, alpha=0.3)

    # Score vs Level
    plt.subplot(2, 4, 6)
    unique_levels = sorted(set(levels))
    avg_scores_by_level = []
    for level in unique_levels:
        level_scores = [scores[i] for i, l in enumerate(levels) if l == level]
        avg_scores_by_level.append(np.mean(level_scores))

    plt.bar(unique_levels, avg_scores_by_level, alpha=0.7, color='cyan')
    plt.title('Average Score by Level')
    plt.xlabel('Level Reached')
    plt.ylabel('Average Score')
    plt.grid(True, alpha=0.3)

    # Learning Progress
    plt.subplot(2, 4, 7)
    periods = [(0, 250, 'Early'), (250, 500, 'Mid'), (500, 750, 'Late'), (750, len(levels), 'Final')]
    period_levels = []
    period_names = []

    for start, end, name in periods:
        if start < len(levels):
            period_data = levels[start:min(end, len(levels))]
            if period_data:
                period_levels.append(np.mean(period_data))
                period_names.append(name)

    if period_levels:
        colors = ['red', 'orange', 'yellow', 'green']
        plt.bar(period_names, period_levels, color=colors[:len(period_levels)], alpha=0.7)
    plt.title('Learning Progress')
    plt.ylabel('Average Level')
    plt.grid(True, alpha=0.3)

    # Final Stats
    plt.subplot(2, 4, 8)
    final_100 = levels[-100:] if len(levels) >= 100 else levels
    stats = [
        ('Avg Level', np.mean(final_100)),
        ('Level 2+%', sum(1 for l in final_100 if l >= 2) / len(final_100) * 100),
        ('Level 3%', sum(1 for l in final_100 if l >= 3) / len(final_100) * 100)
    ]

    names, values = zip(*stats)
    plt.bar(names, values, alpha=0.7, color=['blue', 'green', 'gold'])
    plt.title('Final Performance')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("🎮 Ultra Simple MMORPG - การทดลองครั้งสุดท้าย!")
    print("🎯 เกมง่ายสุดๆ: แผนที่ 5x5, มอนสเตอร์ 3 HP, เลเวลอัพที่ 50 EXP")
    print("=" * 60)

    # ฝึกสอน
    agent, scores, levels, steps_list = train_ultra_simple(1000)

    # แสดงกราฟ
    print("\n📊 แสดงผลการฝึกสอน...")
    plot_ultra_simple_results(scores, levels, steps_list)

    # ทดสอบ
    test_scores, test_levels = evaluate_ultra_simple(agent, 50)

    # Demo การเล่น
    show_agent_play_demo(agent)

    print(f"\n🎉 สรุปผลการทดลอง:")
    print(f"Q-states: {len(agent.q_table)}")
    print(f"Average Level (last 100): {np.mean(levels[-100:]):.2f}")
    print(f"Level 3 Success Rate: {sum(1 for l in levels[-100:] if l >= 3) / 100 * 100:.1f}%")

    if np.mean(levels[-100:]) > 1.5:
        print("🎉 SUCCESS! Agent เรียนรู้ได้แล้ว!")
    else:
        print("😢 ยังไม่สำเร็จ ต้องปรับปรุงเพิ่มเติม")