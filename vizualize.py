import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from map import Map

class SwarmAI(nn.Module):
    def __init__(self):
        super(SwarmAI, self).__init__()
        self.fc1 = nn.Linear(8, 128)  # 21 = 3 (pozycja postaci) + 24 (sąsiedzi)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)  # 4 akcje: góra, dół, lewo, prawo

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MapWithAIVisualization(Map):
    def __init__(self, width: int, height: int, max_characters: int, data_length: int, model, epsilon=0.1):
        super().__init__(width, height, max_characters, data_length)
        self.model = model  # Załaduj wytrenowany model
        self.epsilon = epsilon  # Eksploracja vs Eksploatacja
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.scatters = []  # Lista obiektów scatter dla każdej postaci

    def plot_map(self):
        """
        Funkcja inicjalizująca wykres, w tym tło mapy i postacie.
        """
        # Tworzymy tablicę kolorów, inicjalizowaną na białą (puste pole)
        color_map = np.ones((self.height, self.width, 3))  # Domyślnie białe tło

        # Kolorowanie pól, które są zajęte przez postacie (czarne)
        for character_id in np.where(self.character_active)[0]:
            x, y = self.characters[character_id, 1:3].astype(int)
            color_map[x, y] = [0, 0, 0]  # Czarny kolor dla postaci
        
        # Wyświetlenie mapy (cała mapa jest biała, a zajęte pola czarne)
        self.ax.imshow(color_map, origin='upper', extent=[0, self.width, 0, self.height])
        
        self.ax.set_title('Mapa gry')
        self.ax.set_xlabel('Szerokość')
        self.ax.set_ylabel('Wysokość')
        self.ax.invert_yaxis()  # Odwrócenie osi Y
        self.ax.grid(True, which='both', linestyle='--', color='black', alpha=0.3)

    def update(self, frame):
        """
        Funkcja do aktualizacji mapy w każdej klatce animacji.
        """
        # Wykonaj ruchy dla każdej postaci
        for character_id in np.where(self.character_active)[0]:
            # Przygotowanie danych wejściowych dla modelu
            surroundings = self.get_surroundings(character_id)
            surroundings_flat = surroundings.flatten()
            char_data = self.find_character_by_id(character_id)[3:]  # Wybierz wszystkie dane
            input_data = torch.tensor(np.concatenate([surroundings_flat, char_data]), dtype=torch.float32)

            # Predykcja modelu
            output = self.model(input_data)
            
            # Sprawdź możliwe ruchy
            possible = self.possible_moves(character_id)
            
            # Mnożymy wynik przez możliwe ruchy, aby zignorować niemożliwe
            valid_output = output.detach().numpy() * possible
            move = valid_output.argmax()  # Wybieramy ruch o najwyższej wartości

            # Eksploracja vs. Eksploatacja (epsilon-greedy)
            if np.random.random() < self.epsilon:
                move = np.random.choice([0, 1, 2, 3])  # Losowy ruch

            # Wykonaj ruch tylko, jeśli jest on możliwy
            if move == 0 and possible[0]:  # Góra
                self.move_character(character_id, -1, 0)
            elif move == 1 and possible[1]:  # Dół
                self.move_character(character_id, 1, 0)
            elif move == 2 and possible[2]:  # Lewo
                self.move_character(character_id, 0, -1)
            elif move == 3 and possible[3]:  # Prawo
                self.move_character(character_id, 0, 1)

        # Ponowne rysowanie mapy z nowymi pozycjami postaci
        self.plot_map()

        # Aktualizowanie pozycji postaci na wykresie
        for scatter, character_id in zip(self.scatters, np.where(self.character_active)[0]):
            x, y = self.characters[character_id, 1:3].astype(int)
            scatter.set_offsets([y, x])  # Zmieniamy pozycję punktu na wykresie

        return self.scatters

    def run_animation(self):
        """
        Uruchomienie animacji.
        """
        self.plot_map()  # Inicjalizacja mapy
        ani = FuncAnimation(self.fig, self.update, frames=100, interval=0, repeat=False)
        plt.show()


# Załaduj wytrenowany model
model = SwarmAI()
model.load_state_dict(torch.load('swarm_ai_model.pth'))
model.eval()  # Ustawienie modelu w tryb ewaluacji

# Inicjalizacja mapy z AI
game_map = MapWithAIVisualization(width=5, height=5, max_characters=10, data_length=0, model=model)

# Dodaj postacie
game_map.reset_map()

# Uruchamiamy animację
game_map.run_animation()
