import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Character:
    character_id: int
    x: int
    y: int
    data: np.ndarray


class Map:
    def __init__(self, width: int, height: int, max_characters: int, data_length: int) -> None:
        """
        Tworzy mapę o określonych wymiarach i maksymalnej liczbie postaci.

        :param width: Szerokość mapy.
        :param height: Wysokość mapy.
        :param max_characters: Maksymalna liczba postaci na mapie.
        :param data_length: Rozmiar danych każdej postaci.
        """
        self.width = width
        self.height = height
        self.max_characters = max_characters
        self.data_length = data_length

        # Inicjalizacja mapy i postaci na początku
        self.grid = np.full((width, height), -1, dtype=np.int32)
        self.characters = np.zeros((max_characters, 3 + data_length), dtype=np.float64)  # [character_id, x, y, data...]
        self.character_active = np.zeros(max_characters, dtype=bool)

    def add_character(self, character_id: int, start_x: int, start_y: int, data: np.ndarray) -> None:
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

    def move_character(self, character_id: int, dx: int, dy: int) -> None:
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

    def get_surroundings(self, character_id: int) -> np.ndarray:
        if not self.character_active[character_id]:
            raise ValueError(f"Postać o ID '{character_id}' nie istnieje.")

        x, y = self.characters[character_id, 1:3].astype(int)
        surroundings = np.zeros(8, dtype=int)  # 4 sąsiedzi w czterech kierunkach
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # [góra, dół, lewo, prawo]

        for i, (dx, dy) in enumerate(directions):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.grid[nx, ny] != -1:  # Jeśli pole jest zajęte przez inną postać
                    surroundings[i] = 1

        return surroundings

    def get_all_characters_data(self) -> np.ndarray:
        active_indices = np.where(self.character_active)[0]
        return self.characters[active_indices]

    def find_character_by_id(self, character_id: int) -> np.ndarray:
        if not self.character_active[character_id]:
            raise ValueError(f"Postać o ID '{character_id}' nie istnieje.")
        return self.characters[character_id]

    def get_reward(self) -> int:
        reward = 0  # Początkowa nagroda
        # Sprawdzamy sąsiedztwo aktywnych postaci
        for character_id in np.where(self.character_active)[0]:
            x, y = self.characters[character_id, 1:3].astype(int)
            neighbors = [
                (x - 1, y), (x + 1, y),  # Góra, dół
                (x, y - 1), (x, y + 1)   # Lewo, prawo
            ]
            neighbor_count = 0

            for nx, ny in neighbors:
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[nx, ny] != -1:  # Znaleziono sąsiada
                        neighbor_count += 1
            # Przypisz nagrodę
            if neighbor_count == 0 or neighbor_count == 4:
                reward -= 1
            elif neighbor_count == 2:
                reward += 2
            elif neighbor_count == 3:
                reward += 1

        return reward

    def possible_moves(self, character_id: int) -> List[bool]:
        if not self.character_active[character_id]:
            raise ValueError(f"Postać o ID '{character_id}' nie istnieje.")

        x, y = self.characters[character_id, 1:3].astype(int)
        moves = [False, False, False, False]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # [góra, dół, lewo, prawo]

        for i, (dx, dy) in enumerate(directions):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.grid[nx, ny] == -1:  # Pole jest wolne
                    moves[i] = True

        return moves

    def reset_map(self):
        """
        Funkcja resetująca stan mapy i aktywność postaci,
        przy przygotowaniu mapy do nowego cyklu lub epoki.
        """
        # Resetowanie mapy
        self.grid.fill(-1)

        # Resetowanie danych postaci
        self.characters.fill(0)
        self.character_active.fill(False)

        # Dodawanie nowych postaci na mapę
        for character_id in range(self.max_characters):
            # Generowanie losowej pozycji startowej
            while True:
                start_x = np.random.randint(0, self.width)
                start_y = np.random.randint(0, self.height)
                if self.grid[start_x, start_y] == -1:  # Jeśli pole jest wolne
                    break

            # Generowanie losowych danych dla postaci
            data = np.zeros(self.data_length)  # Przykładowe dane dla postaci
            self.add_character(character_id, start_x, start_y, data)


if __name__ == '__main__':
    import timeit

    # Tworzymy mapę o wymiarach 200x200 z 50 postaciami i 10 danymi na każdą postać
    game_map = Map(width=200, height=200, max_characters=50, data_length=10)

    # Testowanie prędkości resetowania mapy
    print(timeit.timeit(lambda: game_map.reset_map(), number=100))
