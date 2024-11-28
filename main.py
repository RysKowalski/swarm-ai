import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from map import Map

# Model AI
class SwarmAI(nn.Module):
	def __init__(self):
		super(SwarmAI, self).__init__()
		self.fc1 = nn.Linear(24, 128)  # 27 = 3 (pozycja postaci) + 24 (sąsiedzi)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 4)  # 4 akcje: góra, dół, lewo, prawo

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)
		return x


# Pamięć doświadczenia
class ReplayBuffer:
	def __init__(self, capacity=10000):
		self.buffer = deque(maxlen=capacity)

	def add(self, experience):
		self.buffer.append(experience)

	def sample(self, batch_size):
		return random.sample(self.buffer, batch_size)

	def size(self):
		return len(self.buffer)


# Trening modelu z Q-Learning i eksploracją
def train_ai(map: Map, model, replay_buffer, batch_size=64, epsilon=0.1, gamma=0.99, learning_rate=0.001, num_generations=1000):
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	loss_fn = nn.MSELoss()

	# Początkowe zbieranie doświadczeń do replay buffer
	initial_experience_generations = 50  # Liczba generacji do zbierania początkowych doświadczeń
	for generation in range(initial_experience_generations):
		map.reset_map()

		for character_id in range(map.max_characters):
			if not map.character_active[character_id]:
				continue
			
			# Przygotowanie danych wejściowych dla modelu
			surroundings = map.get_surroundings(character_id)
			surroundings_flat = surroundings.flatten()
			char_data = map.find_character_by_id(character_id)[3:6]  # Wybierz tylko 3 elementy
			input_data = torch.tensor(np.concatenate([surroundings_flat, char_data]), dtype=torch.float32)

			# Predykcja modelu
			output = model(input_data)

			# Sprawdź możliwe ruchy
			possible = map.possible_moves(character_id)

			# Mnożymy wynik przez możliwe ruchy, aby zignorować niemożliwe
			valid_output = output.detach().numpy() * possible
			move = valid_output.argmax()  # Wybieramy ruch o najwyższej wartości

			# Eksploracja vs. Eksploatacja (epsilon-greedy)
			if random.random() < epsilon:
				move = np.random.choice([0, 1, 2, 3])  # Losowy ruch

			# Wykonaj ruch tylko, jeśli jest on możliwy
			if move == 0 and possible[0]:  # Góra
				map.move_character(character_id, -1, 0)
			elif move == 1 and possible[1]:  # Dół
				map.move_character(character_id, 1, 0)
			elif move == 2 and possible[2]:  # Lewo
				map.move_character(character_id, 0, -1)
			elif move == 3 and possible[3]:  # Prawo
				map.move_character(character_id, 0, 1)

			# Oblicz nagrodę
			reward = map.get_reward()

			# Zbieramy doświadczenie (stan, akcja, nagroda, nowy stan)
			next_state = torch.tensor(np.concatenate([map.get_surroundings(character_id).flatten(), map.find_character_by_id(character_id)[3:6]]), dtype=torch.float32)
			experience = (input_data, move, reward, next_state)
			replay_buffer.add(experience)

		print(f"Zbieranie doświadczeń: Generacja {generation + 1}/{initial_experience_generations}, Rozmiar bufora: {replay_buffer.size()}")

	# Trening modelu
	for generation in range(initial_experience_generations, num_generations):
		map.reset_map()

		# Zbieranie doświadczeń dla wszystkich postaci
		all_experiences = []
		for character_id in range(map.max_characters):
			if not map.character_active[character_id]:
				continue

			# Wykonaj 100 ruchów dla każdej postaci
			for move_count in range(100):  # 100 ruchów na postać
				# Przygotowanie danych wejściowych dla modelu
				surroundings = map.get_surroundings(character_id)
				surroundings_flat = surroundings.flatten()
				char_data = map.find_character_by_id(character_id)[3:6]  # Wybierz tylko 3 elementy
				input_data = torch.tensor(np.concatenate([surroundings_flat, char_data]), dtype=torch.float32)
				
				# Predykcja modelu
				output = model(input_data)
				
				# Sprawdź możliwe ruchy
				possible = map.possible_moves(character_id)
				
				# Mnożymy wynik przez możliwe ruchy, aby zignorować niemożliwe
				if possible != [False, False, False, False]:
					valid_output = output.detach().numpy() * possible
					move = valid_output.argmax()  # Wybieramy ruch o najwyższej wartości

					# Eksploracja vs. Eksploatacja (epsilon-greedy)
					if random.random() < epsilon:
						move = np.random.choice([0, 1, 2, 3])  # Losowy ruch

					# Wykonaj ruch tylko, jeśli jest on możliwy
					if move == 0 and possible[0]:  # Góra
						map.move_character(character_id, -1, 0)
					elif move == 1 and possible[1]:  # Dół
						map.move_character(character_id, 1, 0)
					elif move == 2 and possible[2]:  # Lewo
						map.move_character(character_id, 0, -1)
					elif move == 3 and possible[3]:  # Prawo
						map.move_character(character_id, 0, 1)

				# Oblicz nagrodę
				reward = map.get_reward()
				
				# Zbieramy doświadczenie (stan, akcja, nagroda, nowy stan)
				next_state = torch.tensor(np.concatenate([map.get_surroundings(character_id).flatten(), map.find_character_by_id(character_id)[3:6]]), dtype=torch.float32)
				experience = (input_data, move, reward, next_state)
				replay_buffer.add(experience)

		# Trening na podstawie doświadczeń
		if replay_buffer.size() > batch_size:
			# Trening na podstawie doświadczeń
			batch = replay_buffer.sample(batch_size)
			for state, action, reward, next_state in batch:
				# Q-Learning Update
				target = reward + gamma * model(next_state).max().item()
				output = model(state)
				target_f = output.clone()
				target_f[action] = target

				# Obliczanie straty
				loss = loss_fn(output, target_f)

				# Optymalizacja
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			print(f"Generacja {generation + 1}/{num_generations}, Strata: {loss.item()}, reward: {reward}")
		else:
			print(f"Generacja {generation + 1}/{num_generations}, Brak wystarczającej liczby próbek w replay bufferze")

		# Zmniejsz epsilon, aby zwiększyć eksplorację
		if epsilon > 0.01:
			epsilon *= 0.995

		# Zapisz model po zakończeniu treningu
		if generation == num_generations - 1:
			torch.save(model.state_dict(), 'swarm_ai_model.pth')
			print("Model zapisany!")


# Tworzenie instancji mapy i modelu
map = Map(200, 200, 50, 0)  # Mapa 2000x2000, 50 postaci, dane postaci o długości 3
model = SwarmAI()

# Pamięć doświadczenia
replay_buffer = ReplayBuffer()

# Trenujemy model
train_ai(map, model, replay_buffer)
