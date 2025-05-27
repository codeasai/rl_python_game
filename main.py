import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import json
import os
from datetime import datetime
from collections import defaultdict


class PersistentQLearningAgent:
    """Agent ที่สามารถ Save/Load ความรู้ได้"""

    def __init__(self, save_path="models/", agent_name="mmorpg_agent"):
        self.action_size = 5
        self.learning_rate = 0.3
        self.discount_factor = 0.95
        self.epsilon = 0.8
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.1

        # Q-table และข้อมูลการเรียนรู้
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))
        self.training_history = {
            'episodes_trained': 0,
            'total_training_time': 0,
            'scores': [],
            'levels': [],
            'training_sessions': []
        }

        # การจัดการไฟล์
        self.save_path = save_path
        self.agent_name = agent_name
        self.model_file = os.path.join(save_path, f"{agent_name}.pkl")
        self.history_file = os.path.join(save_path, f"{agent_name}_history.json")

        # สร้างโฟลเดอร์ถ้ายังไม่มี
        os.makedirs(save_path, exist_ok=True)

        print(f"🤖 Persistent Agent initialized: {agent_name}")
        print(f"📁 Save path: {save_path}")

    def save_agent(self):
        """บันทึก Agent (Q-table และประวัติ)"""
        try:
            # บันทึก Q-table
            with open(self.model_file, 'wb') as f:
                # แปลง defaultdict เป็น dict ปกติ
                q_table_dict = dict(self.q_table)
                agent_data = {
                    'q_table': q_table_dict,
                    'epsilon': self.epsilon,
                    'episodes_trained': self.training_history['episodes_trained'],
                    'save_timestamp': datetime.now().isoformat()
                }
                pickle.dump(agent_data, f)

            # บันทึกประวัติ (JSON สำหรับอ่านง่าย)
            with open(self.history_file, 'w') as f:
                json.dump(self.training_history, f, indent=2)

            print(f"💾 Agent saved successfully!")
            print(f"📊 Episodes trained: {self.training_history['episodes_trained']}")
            print(f"🧠 Q-states learned: {len(self.q_table)}")

        except Exception as e:
            print(f"❌ Error saving agent: {e}")

    def load_agent(self):
        """โหลด Agent (Q-table และประวัติ)"""
        try:
            # โหลด Q-table
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    agent_data = pickle.load(f)

                # แปลงกลับเป็น defaultdict
                self.q_table = defaultdict(lambda: np.zeros(self.action_size))
                for state, q_values in agent_data['q_table'].items():
                    self.q_table[state] = np.array(q_values)

                self.epsilon = agent_data.get('epsilon', self.epsilon)
                episodes = agent_data.get('episodes_trained', 0)
                save_time = agent_data.get('save_timestamp', 'Unknown')

                print(f"✅ Q-table loaded successfully!")
                print(f"📅 Saved on: {save_time}")
                print(f"📊 Episodes trained: {episodes}")
                print(f"🧠 Q-states loaded: {len(self.q_table)}")
                print(f"🎯 Current epsilon: {self.epsilon:.3f}")

            else:
                print(f"ℹ️  No saved model found. Starting fresh!")
                return False

            # โหลดประวัติ
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.training_history = json.load(f)
                print(f"📈 Training history loaded!")

            return True

        except Exception as e:
            print(f"❌ Error loading agent: {e}")
            return False

    def get_action(self, state):
        """เลือก action (เหมือนเดิม)"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        """เรียนรู้ (เหมือนเดิม)"""
        current_q = self.q_table[state][action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])

        self.q_table[state][action] += self.learning_rate * (target_q - current_q)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def add_training_session(self, episodes, scores, levels):
        """เพิ่มข้อมูลการฝึกสอนครั้งใหม่"""
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'episodes': episodes,
            'avg_score': float(np.mean(scores)),
            'avg_level': float(np.mean(levels)),
            'final_epsilon': float(self.epsilon),
            'q_states_count': len(self.q_table)
        }

        self.training_history['episodes_trained'] += episodes
        self.training_history['scores'].extend(scores)
        self.training_history['levels'].extend(levels)
        self.training_history['training_sessions'].append(session_data)

    def show_training_summary(self):
        """แสดงสรุปการฝึกสอนทั้งหมด"""
        print(f"\n📊 === Training Summary ===")
        print(f"🤖 Agent: {self.agent_name}")
        print(f"📚 Total episodes: {self.training_history['episodes_trained']}")
        print(f"🧠 Q-states learned: {len(self.q_table)}")
        print(f"🎯 Current epsilon: {self.epsilon:.3f}")

        if self.training_history['training_sessions']:
            print(f"\n📈 Training Sessions:")
            for i, session in enumerate(self.training_history['training_sessions']):
                print(f"  Session {i + 1}: {session['episodes']} episodes, "
                      f"Avg Score: {session['avg_score']:.1f}, "
                      f"Avg Level: {session['avg_level']:.2f}")

        # แสดงประสิทธิภาพล่าสุด
        if len(self.training_history['levels']) >= 100:
            recent_levels = self.training_history['levels'][-100:]
            recent_scores = self.training_history['scores'][-100:]
            level_3_rate = sum(1 for l in recent_levels if l >= 3) / 100 * 100

            print(f"\n🎯 Recent Performance (last 100 episodes):")
            print(f"  Average Level: {np.mean(recent_levels):.2f}")
            print(f"  Average Score: {np.mean(recent_scores):.1f}")
            print(f"  Level 3 Success Rate: {level_3_rate:.1f}%")


class UltraSimpleMMORPGEnv:
    """Environment เดิม (ไม่เปลี่ยนแปลง)"""

    def __init__(self):
        self.player_hp = 100
        self.player_level = 1
        self.player_exp = 0
        self.player_x = 2
        self.player_y = 2
        self.map_size = 5
        self.monster_x = 1
        self.monster_y = 1
        self.monster_hp = 3
        self.monster_max_hp = 3
        self.action_size = 5

    def get_state(self):
        distance_to_monster = abs(self.player_x - self.monster_x) + abs(self.player_y - self.monster_y)
        state = [
            min(3, self.player_level - 1),
            min(3, self.player_exp // 25),
            min(4, distance_to_monster),
            min(2, self.monster_hp)
        ]
        return tuple(state)

    def step(self, action):
        reward = 0
        done = False
        info = {}

        if action == 0 and self.player_y > 0:
            self.player_y -= 1
            reward += 0.1
        elif action == 1 and self.player_y < self.map_size - 1:
            self.player_y += 1
            reward += 0.1
        elif action == 2 and self.player_x < self.map_size - 1:
            self.player_x += 1
            reward += 0.1
        elif action == 3 and self.player_x > 0:
            self.player_x -= 1
            reward += 0.1
        elif action == 4:
            attack_reward = self._attack_monster()
            reward += attack_reward

        if self.player_exp >= 50:
            self.player_level += 1
            self.player_exp = 0
            reward += 100
            print(f"🎉 LEVEL UP! Now level {self.player_level}")
            self._spawn_new_monster()

        if self.player_level >= 3:
            done = True
            reward += 200
            print("🏆 YOU WIN!")

        info['level'] = self.player_level
        info['exp'] = self.player_exp

        return self.get_state(), reward, done, info

    def _attack_monster(self):
        distance = abs(self.player_x - self.monster_x) + abs(self.player_y - self.monster_y)

        if distance > 1:
            return -1
        if self.monster_hp <= 0:
            return -2

        self.monster_hp -= 1

        if self.monster_hp <= 0:
            exp_gained = 30
            self.player_exp += exp_gained
            print(f"💀 Monster defeated! +{exp_gained} EXP")
            self._spawn_new_monster()
            return 50
        else:
            return 10

    def _spawn_new_monster(self):
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
        self.monster_x = 1
        self.monster_y = 1
        self.monster_hp = self.monster_max_hp
        return self.get_state()


def train_or_continue_agent(agent_name="smart_mmorpg_agent", new_episodes=500):
    """ฝึกสอน Agent ใหม่หรือฝึกต่อจากเดิม"""

    print(f"🎮 === Training Session ===")
    print(f"🤖 Agent: {agent_name}")
    print(f"📚 Episodes to train: {new_episodes}")

    # สร้าง Agent
    agent = PersistentQLearningAgent(agent_name=agent_name)

    # พยายามโหลด Agent เดิม
    loaded = agent.load_agent()
    if loaded:
        print(f"🔄 Continuing training from saved model...")
    else:
        print(f"🆕 Starting fresh training...")

    # แสดงสรุปก่อนเริ่มฝึก
    agent.show_training_summary()

    # ฝึกสอน
    env = UltraSimpleMMORPGEnv()
    scores = []
    levels = []

    print(f"\n🏃‍♂️ Starting training for {new_episodes} episodes...")

    for episode in range(new_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 50

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
        levels.append(info['level'])

        # รายงานผลทุก 100 episodes
        if (episode + 1) % 100 == 0:
            recent_scores = scores[-100:]
            recent_levels = levels[-100:]
            avg_score = np.mean(recent_scores)
            avg_level = np.mean(recent_levels)
            level_3_rate = sum(1 for l in recent_levels if l >= 3) / len(recent_levels) * 100

            print(f"Episode {episode + 1:3d} | Score: {avg_score:6.1f} | "
                  f"Level: {avg_level:.2f} | Level3: {level_3_rate:4.1f}% | ε: {agent.epsilon:.3f}")

    # บันทึกผลการฝึกสอน
    agent.add_training_session(new_episodes, scores, levels)
    agent.save_agent()

    print(f"\n✅ Training completed!")
    agent.show_training_summary()

    return agent, scores, levels


def test_saved_agent(agent_name="smart_mmorpg_agent", test_episodes=50):
    """ทดสอบ Agent ที่บันทึกไว้"""

    print(f"\n🧪 === Testing Saved Agent ===")

    # โหลด Agent
    agent = PersistentQLearningAgent(agent_name=agent_name)
    if not agent.load_agent():
        print(f"❌ No saved agent found!")
        return

    # ปิด exploration
    agent.epsilon = 0

    env = UltraSimpleMMORPGEnv()
    scores = []
    levels = []

    print(f"🎯 Testing for {test_episodes} episodes (ε=0)...")

    for episode in range(test_episodes):
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

    # แสดงผลการทดสอบ
    avg_score = np.mean(scores)
    avg_level = np.mean(levels)
    level_3_rate = sum(1 for l in levels if l >= 3) / len(levels) * 100
    level_2_rate = sum(1 for l in levels if l >= 2) / len(levels) * 100

    print(f"\n📊 === Test Results ===")
    print(f"🎯 Average Score: {avg_score:.2f}")
    print(f"⭐ Average Level: {avg_level:.2f}")
    print(f"📈 Level 2+ Rate: {level_2_rate:.1f}%")
    print(f"🏆 Level 3 Rate: {level_3_rate:.1f}%")

    return scores, levels


def compare_fresh_vs_saved():
    """เปรียบเทียบ Agent ใหม่ vs Agent ที่บันทึกไว้"""

    print(f"\n⚔️  === Fresh vs Saved Agent Comparison ===")

    # ตรวจสอบว่ามี Saved Agent หรือไม่
    temp_agent = PersistentQLearningAgent(agent_name="smart_mmorpg_agent")
    has_saved = temp_agent.load_agent()

    if not has_saved:
        print(f"⚠️ ไม่พบ Saved Agent! กรุณาฝึก Agent ก่อน")
        print(f"💡 วิธีแก้: เลือก option 1 เพื่อฝึก Agent ก่อน")
        return

    # ทดสอบ Fresh Agent
    print(f"\n1️⃣ Testing Fresh Agent (500 episodes)...")
    fresh_agent = PersistentQLearningAgent(agent_name="temp_fresh_agent")
    env = UltraSimpleMMORPGEnv()

    fresh_scores = []
    for episode in range(500):
        state = env.reset()
        total_reward = 0
        steps = 0

        while steps < 50:
            action = fresh_agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            fresh_agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        fresh_scores.append(total_reward)

        # แสดงความคืบหน้า
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(fresh_scores[-100:])
            print(f"  Episode {episode + 1}: Avg Score = {avg_score:.1f}")

    fresh_avg = np.mean(fresh_scores[-100:])

    # ทดสอบ Saved Agent
    print(f"\n2️⃣ Testing Saved Agent...")
    saved_scores, saved_levels = test_saved_agent("smart_mmorpg_agent")
    saved_avg = np.mean(saved_scores) if saved_scores else 0

    # เปรียบเทียบ
    print(f"\n📊 === Comparison Results ===")
    print(f"🆕 Fresh Agent (500 episodes): {fresh_avg:.2f}")
    print(f"💾 Saved Agent (immediate):    {saved_avg:.2f}")

    if saved_avg > fresh_avg:
        improvement = saved_avg - fresh_avg
        print(f"✅ Saved agent is {improvement:.1f} points better!")
        print(f"⏰ Saved {500} episodes of training time!")
    else:
        print(f"⚠️ Fresh agent performed better. Consider retraining.")


def demo_save_load_workflow():
    """แสดงตัวอย่างการใช้งาน Save/Load"""

    print(f"🎓 === Save/Load Workflow Demo ===")
    print(f"1. Train new agent")
    print(f"2. Save agent")
    print(f"3. Test saved agent")
    print(f"4. Continue training")
    print(f"5. Compare performance")

    # 1. ฝึก Agent ใหม่
    agent, scores, levels = train_or_continue_agent("demo_agent", 300)

    # 2. ทดสอบ Agent ที่บันทึก
    test_saved_agent("demo_agent")

    # 3. ฝึกต่อเพิ่มอีก
    print(f"\n🔄 Continue training for 200 more episodes...")
    train_or_continue_agent("demo_agent", 200)

    # 4. ทดสอบอีกครั้ง
    test_saved_agent("demo_agent")


if __name__ == "__main__":
    print(f"🎮 Persistent RL Agent System")
    print(f"=" * 50)

    choice = input("""
เลือกการทำงาน:
1. ฝึก Agent ใหม่หรือฝึกต่อจากเดิม
2. ทดสอบ Agent ที่บันทึกไว้
3. เปรียบเทียบ Fresh vs Saved
4. Demo การใช้งานทั้งหมด

เลือก (1-4): """)

    if choice == "1":
        episodes = int(input("จำนวน episodes ที่ต้องการฝึก (500): ") or "500")
        train_or_continue_agent("smart_mmorpg_agent", episodes)

    elif choice == "2":
        test_saved_agent("smart_mmorpg_agent")

    elif choice == "3":
        compare_fresh_vs_saved()

    elif choice == "4":
        demo_save_load_workflow()

    else:
        print("ใช้ default: ฝึก Agent 500 episodes")
        train_or_continue_agent("smart_mmorpg_agent", 500)
