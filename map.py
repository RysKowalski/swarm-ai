import numpy as np
from dataclasses import dataclass
from typing import Self

@dataclass
class Character:
	character_id: int
	x: int
	y: int
	data: np.ndarray


class Map:
	def __init__(self: Self, width: int, height: int, max_characters: int, data_length: int) -> None:
		"""
		Tworzy mapę o określonych wymiarach i maksymalnej liczbie postaci.

		:param width: Szerokość mapy.
		:param height: Wysokość mapy.
		:param max_characters: Maksymalna liczba postaci na mapie.
		:param data_length: Rozmiar danych każdej postaci.
		"""
		self.width: int = width
		self.height: int = height
		self.max_characters: int = max_characters
		self.data_length: int = data_length

		# Mapa zawierająca ID postaci (lub -1 oznaczające brak postaci)
		self.grid: np.ndarray = np.full((width, height), -1, dtype=np.int32)

		# Dane wszystkich postaci
		self.characters: np.ndarray = np.zeros(
			(max_characters, 3 + data_length), dtype=np.float64
		)  # [character_id, x, y, data...]

		# Maska śledząca, czy slot w `self.characters` jest zajęty
		self.character_active: np.ndarray = np.zeros(max_characters, dtype=bool)

	def add_character(self: Self, character_id: int, start_x: int, start_y: int, data: np.ndarray) -> None:
		if self.character_active[character_id]:
			raise ValueError(f"Postać o ID '{character_id}' już istnieje.")
		if not (0 <= start_x < self.width and 0 <= start_y < self.height):
			raise ValueError("Pozycja początkowa poza granicami mapy.")
		if self.grid[start_x, start_y] != -1:
			raise ValueError(f"Pole ({start_x}, {start_y}) jest już zajęte.")
		if len(data) != self.data_length:
			raise ValueError(f"Dane muszą mieć długość {self.data_length}.")

		# Dodaj postać
		self.grid[start_x, start_y] = character_id
		self.characters[character_id, 0] = character_id
		self.characters[character_id, 1] = start_x
		self.characters[character_id, 2] = start_y
		self.characters[character_id, 3:] = data
		self.character_active[character_id] = True

	def move_character(self: Self, character_id: int, dx: int, dy: int) -> None:
		if not self.character_active[character_id]:
			raise ValueError(f"Postać o ID '{character_id}' nie istnieje.")

		x, y = self.characters[character_id, 1:3].astype(int)
		new_x = x + dx
		new_y = y + dy

		# Sprawdzenie granic mapy
		if not (0 <= new_x < self.width and 0 <= new_y < self.height):
			raise ValueError(f"Ruch poza granice mapy: ({new_x}, {new_y}).")

		if self.grid[new_x, new_y] != -1:
			raise ValueError(f"Pole ({new_x}, {new_y}) jest już zajęte.")

		# Przenieś postać
		self.grid[x, y] = -1
		self.grid[new_x, new_y] = character_id
		self.characters[character_id, 1:3] = [new_x, new_y]

	def get_surroundings(self: Self, character_id: int) -> np.ndarray:
		if not self.character_active[character_id]:
			raise ValueError(f"Postać o ID '{character_id}' nie istnieje.")

		x, y = self.characters[character_id, 1:3].astype(int)
		surroundings = np.zeros((5, 5), dtype=int)  # Twórz zawsze siatkę 5x5 z zerami
		x_min, x_max = max(0, x - 2), min(self.width, x + 3)
		y_min, y_max = max(0, y - 2), min(self.height, y + 3)

		surrounding_ids = self.grid[x_min:x_max, y_min:y_max]

		for i in range(surrounding_ids.shape[0]):
			for j in range(surrounding_ids.shape[1]):
				char_id = surrounding_ids[i, j]
				if char_id != -1:  # Jeśli pole jest pełne
					surroundings[i, j] = 1

		# Usuń środkowy element
		surroundings = np.delete(surroundings.flatten(), 12)
		return surroundings

	def get_all_characters_data(self) -> np.ndarray:
		active_indices = np.where(self.character_active)[0]
		return self.characters[active_indices]

	def find_character_by_id(self: Self, character_id: int) -> np.ndarray:
		if not self.character_active[character_id]:
			raise ValueError(f"Postać o ID '{character_id}' nie istnieje.")
		return self.characters[character_id]

	def get_reward(self: Self) -> int:
		"""
		Oblicza nagrodę na podstawie pozycji postaci.
		Zaczyna od 0 punktów i odejmuje 100 za każdą postać, która
		bezpośrednio obok siebie nie ma żadnej innej postaci.

		:return: Łączna nagroda.
		"""
		reward = 0  # Początkowa nagroda

		# Przechodzimy przez wszystkie aktywne postacie
		for character_id in np.where(self.character_active)[0]:
			x, y = self.characters[character_id, 1:3].astype(int)

			# Sprawdzamy sąsiadujące pola (góra, dół, lewo, prawo)
			neighbors = [
				(x - 1, y), (x + 1, y),  # Góra, dół
				(x, y - 1), (x, y + 1)   # Lewo, prawo
			]
			neighbor_count = 0

			for nx, ny in neighbors:
				if 0 <= nx < self.width and 0 <= ny < self.height:
					if self.grid[nx, ny] != -1:  # Znaleziono sąsiada
						neighbor_count += 1
						

			# Jeśli nie ma żadnego sąsiada, odejmujemy 100 punktów
			if neighbor_count == 0 or neighbor_count == 4:
				reward -= 1
			elif neighbor_count == 2:
				reward += 2
			elif neighbor_count == 3:
				reward += 1
			

		return reward
	
	def possible_moves(self: Self, character_id: int) -> list[bool]:
		"""
		Sprawdza możliwe ruchy dla postaci w czterech kierunkach, uwzględniając, czy pole jest wolne.

		:param character_id: ID postaci.
		:return: Lista czterech wartości logicznych [góra, dół, lewo, prawo].
		"""
		if not self.character_active[character_id]:
			raise ValueError(f"Postać o ID '{character_id}' nie istnieje.")

		x, y = self.characters[character_id, 1:3].astype(int)
		
		# Lista ruchów: [góra, dół, lewo, prawo]
		moves = [False, False, False, False]

		directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # [góra, dół, lewo, prawo]

		for i, (dx, dy) in enumerate(directions):
			nx, ny = x + dx, y + dy
			if 0 <= nx < self.width and 0 <= ny < self.height:
				# Sprawdzenie, czy pole nie jest zajęte przez inną postać
				if self.grid[nx, ny] == -1:  # Pole jest wolne
					moves[i] = True

		return moves
	
	def reset_map(self):
		"""
		Funkcja resetująca stan mapy i aktywność postaci,
		przy przygotowaniu mapy do nowego cyklu lub epoki.
		"""
		# Resetowanie pozycji na mapie (wszystkie wartości w grid do -1)
		self.grid.fill(-1)

		# Resetowanie danych postaci (wygaszenie wszystkich postaci i ich danych)
		self.characters.fill(0)
		self.character_active.fill(False)

		# Dodawanie nowych postaci na mapę
		for character_id in range(self.max_characters):
			# Generowanie losowej pozycji startowej
			while True:
				start_x = np.random.randint(0, self.width)
				start_y = np.random.randint(0, self.height)

				# Sprawdzanie, czy pole jest wolne
				if self.grid[start_x, start_y] == -1:
					break  # Jeśli pole jest wolne, wychodzimy z pętli

			# Generowanie losowych danych dla postaci (np. losowe wartości w tablicy)
			data = np.random.rand(self.data_length)  # Przykład generowania losowych danych (np. 3 dane)

			# Dodawanie postaci na mapę
			self.add_character(character_id, start_x, start_y, data)
			#print(f"Postać {character_id} dodana na pozycję ({start_x}, {start_y}) z danymi: {data}")
