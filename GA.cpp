#include <iostream>
#include <fstream>
#include <functional>
#include <algorithm>
#include <random>
#include <csignal>

#include <chrono>
#include <thread>

#include "../Neural Network/nn.cpp"

using std::function,
std::literals::chrono_literals::operator""s,
std::chrono::steady_clock;

bool sign = true;
bool dump = false;
float mutateChance = 0.1f;
float crossChance = 0.7f;

class Population {
  public:
    vector<NeuralNetwork> individs{};

    Population(int size, vector<vector<int>> sizes) {
      this->individs.resize(size);

      for (int individ = 0; individ < size; individ++) {

        vector<Matrix<float>> layers(sizes.size());
        vector<Matrix<float>> biases(sizes.size());

        for (int L = 0; L < sizes.size(); L++) {
          int rowSize = sizes[L][0], colSize = sizes[L][1];
          vector<vector<float>> rows(rowSize);

          for (int row = 0; row < rowSize; row++) {
            rows[row] = vector<float>(colSize);
            for (int col = 0; col < colSize; col++) {
              rows[row][col] = (float) rand() / RAND_MAX;
            }
          }
          layers[L] = Matrix<float>(rows);

          //colSize because height of second matrix must be equal to width of first for multiplying
          for (int row = 0; row < colSize; row++) {
            rows[row] = vector<float> {
              (float) rand() / RAND_MAX
            };
          }
          biases[L] = Matrix<float>(rows);
        }

        this->individs[individ] = NeuralNetwork(layers, biases);
      }
    }

    NeuralNetwork& operator[] (int index) {
      return individs[index];
    }

    int size() {
      return individs.size();
    }

    void doSelection(function<bool (NeuralNetwork, NeuralNetwork)> fitness) {
      vector<NeuralNetwork> offspring(this->individs.size());

      for (int i = 0; i < offspring.size(); i++) {
        vector<NeuralNetwork> pretenders{};


        std::sample(
          this->individs.begin(), this->individs.end(),
          std::back_inserter(pretenders),
          2,
          std::mt19937{std::random_device{}()}
        );

        if (fitness(pretenders[0], pretenders[1])) {
          offspring[i] = pretenders[1];
        }
        else {
          offspring[i] = pretenders[0];
        }
      }
      individs = std::move(offspring);
      return;
    }

    int mutate(float probability) {
      int count = 0;

      for (int i = 0; i < this->individs.size(); i++) {
        if ((float) rand() / RAND_MAX > probability) {
          continue;
        }

        NeuralNetwork individ(individs[i].layers, individs[i].biases);
        Matrix<float> layer = individ[rand() % (individ.size() - 2)];
        layer
        [rand() % (layer.height() - 1)]
        [rand() % (layer.width() - 1)] = float(-1000 + rand() % 1000) / 1000.0f;

        count++;
      }

      return count;
    }

    int cross(float probability) {
      int count = 0;

      for (int i = 0; i < this->individs.size() / 2; i++) {
        if ((float) rand() / RAND_MAX > probability) {
          continue;
        }

        NeuralNetwork parent1 = individs[i], parent2 = individs[i + 1],
        child1 = parent1, child2 = parent2;
        int size = parent1.size();

        int slice = 1 + rand() % (size - 2);

        for (int L = slice; L < size; L++) {
          child1[L] = parent2[L];
          child2[L] = parent1[L];
        }

        individs[i] = child1;
        individs[i + 1] = child2;

        count++;
      }

      return count;
    }
};


bool fitness(NeuralNetwork first, NeuralNetwork second) {
  bool turn = true;
  bool are_playing = true;
  Matrix<float> field(
    {
      {0},
      {0},
      {0},
      {0},
      {0},
      {0},
      {0},
      {0},
      {0}
    }
  );
  Matrix<bool> binField(
    {
      {0, 0, 0},
      {0, 0, 0},
      {0, 0, 0}
    }
  );
  Matrix<bool> transposedField;
  vector<bool> win {
    true, true, true
  };
  vector<bool> diagonal{
    false, false, false
  };
  vector<bool> otherDiagonal{
    false, false, false
  };
  vector<int> skip;
  Matrix<float> output;
  NeuralNetwork player;

  do {
    if (turn) {
      player = first;
    }
    else {
      player = second;
    }

    output = player.getOutput(field);

    float max = 0;
    int maxRow = 0;

    for (int row = 0; row < 9; row++) {
      if (std::find(skip.begin(), skip.end(), row) != skip.end()) {
        continue;
      }
      if (output[row][0] > max) {
        max = output[row][0];
        maxRow = row;
      }
    }

    field[maxRow][0] = 1;
    skip.push_back(maxRow);

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        if (field[3 * i + j][0] == 1) {
          binField[i][j] = true;
        }
        else {
          binField[i][j] = false;
        }
      }
    }

    transposedField = binField.getTransposed();

    for (int i = 0; i < 3; i++) {
      diagonal[i] = binField[i][i];
      otherDiagonal[i] = binField[i][2 - i];
    }

    if (std::find(binField.rows.begin(), binField.rows.end(), win) != binField.rows.end()
         || std::find(transposedField.rows.begin(), transposedField.rows.end(), win) != transposedField.rows.end()
         || diagonal == win
         || otherDiagonal == win
         || skip.size() == 9) {
      are_playing = false;
    }

    field = field * -1;
    turn = !turn;

  } while (are_playing);

  return turn;
}

int main(int argc, char *argv[]) {
  int c = 0;
  Population pop(
    100,
    {
      {9, 9},
      {9, 9},
      {9, 9},
      {9, 9},
      {9, 9},
      {9, 9},
      {9, 9},
      {9, 9},
      {9, 9}
    }
  );
  signal(
    SIGINT,
    [](int p) {
      std::string s;
      std::cout << "Settings\nd - dump, s - stop, c - cross chance, m - mutate chance" << std::endl;
      std::cin >> s;
      if (s == "d") {
        dump = true;
      }
      else if (s == "s") {
        sign = false;
      }
      else if (s == "c") {
        std::cin >> crossChance;
      }
      else if (s == "m") {
        std::cin >> mutateChance;
      }
    }
  );

  do {
    pop.doSelection(fitness);
    system("clear");

    std::cout << "Gen: " << ++c << std::endl << std::endl;

    std::cout << "Crossed: " << pop.cross(crossChance) << std::endl;
    std::cout << "Mutated: " << pop.mutate(mutateChance) << std::endl;

    if (!(c % 2500) or dump) {
      std::this_thread::sleep_for(1s);
      std::ofstream file("data/" + std::to_string(c) + ".json");
      std::sort(pop.individs.begin(), pop.individs.end(), fitness);
      NeuralNetwork best = pop.individs[pop.individs.size() - 1];
      file << best;
      dump = false;
    }
  } while (sign);

  std::ofstream file(std::to_string(c) + ".json");
  std::sort(pop.individs.begin(), pop.individs.end(), fitness);
  NeuralNetwork best = pop.individs[pop.individs.size() - 1];
  file << best;

  
}
