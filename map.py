import numpy as np
from typing import Self

class Character:
    def __init__(self: Self, character_id: int, start_x: int, start_y: int, data: np.ndarray) -> None:
        """
        Inicjalizuje postać z unikalnym ID, pozycją początkową i dodatkowymi danymi.
        
        :param character_id: Unikalne ID postaci (int).
        :param start_x: Pozycja startowa X.
        :param start_y: Pozycja startowa Y.
        :param data: Dane związane z postacią (dowolne w formacie np.ndarray).
        """
        self.character_id: int = character_id
        self.x: int = start_x
        self.y: int = start_y
        self.data: np.ndarray = data  # np.ndarray, reprezentujące dane postaci

class Map:
    def __init__(self: Self, width: int, height: int) -> None:
        """
        Tworzy mapę o określonych wymiarach.
        
        :param width: Szerokość mapy.
        :param height: Wysokość mapy.
        """
        self.width: int = width
        self.height: int = height
        self.characters: dict = {}  # Słownik przechowujący postacie według ich ID
        # Tablica przechowująca referencje do postaci lub None
        self.grid = np.full((width, height), None, dtype=object)

    def add_character(self: Self, character_id: int, start_x: int, start_y: int, data: np.ndarray) -> None:
        """
        Dodaje nową postać na mapę.
        
        :param character_id: Unikalne ID postaci (int).
        :param start_x: Pozycja startowa X.
        :param start_y: Pozycja startowa Y.
        :param data: Dane związane z postacią.
        """
        if character_id in self.characters:
            raise ValueError(f"Postać o ID '{character_id}' już istnieje.")
        if not (0 <= start_x < self.width and 0 <= start_y < self.height):
            raise ValueError("Pozycja początkowa poza granicami mapy.")
        if self.grid[start_x, start_y] is not None:
            raise ValueError(f"Pole ({start_x}, {start_y}) jest już zajęte.")

        character: Character = Character(character_id, start_x, start_y, data)
        self.characters[character_id] = character
        self.grid[start_x, start_y] = character

    def move_character(self: Self, character_id: int, dx: int, dy: int) -> None:
        """
        Porusza postać o podanym ID w określonym kierunku.
        
        :param character_id: ID postaci (int).
        :param dx: Zmiana współrzędnej X.
        :param dy: Zmiana współrzędnej Y.
        """
        if character_id not in self.characters:
            raise ValueError(f"Postać o ID '{character_id}' nie istnieje.")
        
        character: Character = self.characters[character_id]
        new_x = character.x + dx
        new_y = character.y + dy

        # Sprawdź granice mapy
        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            if self.grid[new_x, new_y] is not None:
                raise ValueError(f"Pole ({new_x}, {new_y}) jest już zajęte.")
            
            # Aktualizuj pozycję
            self.grid[character.x, character.y] = None
            character.x = new_x
            character.y = new_y
            self.grid[new_x, new_y] = character
        else:
            print("Ruch poza granice mapy!")

    def get_surroundings(self: Self, character_id: int) -> np.ndarray:
        """
        Zwraca otoczenie 2 pikseli dookoła postaci o podanym ID.
        
        :param character_id: ID postaci (int).
        :return: np.ndarray zawierający dane postaci lub None.
        """
        if character_id not in self.characters:
            raise ValueError(f"Postać o ID '{character_id}' nie istnieje.")
        
        character = self.characters[character_id]
        x, y = character.x, character.y

        # Wyznacz granice otoczenia
        x_min = max(0, x - 2)
        x_max = min(self.width, x + 3)
        y_min = max(0, y - 2)
        y_max = min(self.height, y + 3)

        # Wytnij otoczenie z siatki mapy
        surroundings = self.grid[x_min:x_max, y_min:y_max]
        return np.vectorize(lambda obj: obj.data if obj else None)(surroundings)

# Przykład użycia
if __name__ == "__main__":
    # Tworzymy mapę o rozmiarze 10x10
    game_map = Map(10, 10)

    # Dodajemy postacie z różnymi danymi
    game_map.add_character(1, 5, 5, np.array([100, 50]))  # np.ndarray reprezentujące np. zdrowie i moc
    game_map.add_character(2, 3, 3, np.array([70, 30]))

    # Poruszamy postać
    game_map.move_character(1, 0, 1)

    # Pobieramy otoczenie
    print("Otoczenie postaci 1:")
    print(game_map.get_surroundings(1))

    # Wyświetlamy otoczenie drugiej postaci
    print("Otoczenie postaci 2:")
    print(game_map.get_surroundings(2))
