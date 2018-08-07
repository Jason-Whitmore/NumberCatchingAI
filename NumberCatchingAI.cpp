#include "NumberCatchingAI.h"


NumberCatchingAI::NumberCatchingAI(){
    srand(time(0));
    int numCols = 15;
    int numRows = 25;

    playerLocation = 8;

    std::vector<char> newRow;

    for(int r = 0; r < numRows; r++){
        newRow = std::vector<char>();
        for(int c = 0; c < numCols; c++){
            newRow.push_back(' ');
        }

        environment.push_back(newRow);
    }

    environment[numRows - 1][playerLocation] = 'U';

    int colLocation;
    int value;
    NumberRecord n;
    for(int row = 4; row < 25; row += 5){
        colLocation = getRandomInt(0, 15);
        value = getRandomInt(1,9);
        environment[row][colLocation] = '0' + value;
        n.column = colLocation;
        n.row = row;
        n.value = value;

        numbers.emplace(n);
    }


}

void NumberCatchingAI::printGame(){
    for(int r = 0; r < environment.size(); r++){
        //std::cout << "-------------------------------" << std::endl;
        for(int c = 0; c < environment[0].size(); c++){
            std::cout << "|" << environment[r][c];
        }

        std::cout << "|\n";
    }

    std::cout << "-------------------------------" << std::endl;
}

int NumberCatchingAI::getRandomInt(int min, int max){


    if(min == max){
        return min;
    }

    return min + (rand() % (max - min));
}




int main(){
    std::cout << "Hello world!" << std::endl;
    NumberCatchingAI n = NumberCatchingAI();

    n.printGame();
    
}