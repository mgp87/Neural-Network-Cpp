#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>

using namespace std;

typedef vector<int> arr1d;
typedef vector<arr1d> arr3d;
vector<arr3d> outputs;

int main(){
    for(int i = 0; i < 255; i++){
        outputs.push_back(arr3d());
    }

    for(int r = 0; r < 255; r+=10){
        for(int g = 1; g <= 255; g+=10){
            for(int b = 2; b <= 255; b+=10){
                int input = 0.63*r + 0.21*g + 0.14*b;
                if(outputs[input].size() <= 5 && input <= 255){
                    arr1d temp;
                    temp.push_back(r);
                    temp.push_back(g);
                    temp.push_back(b);
                    outputs[input].push_back(temp);
                }
            }
        }
    }

    for(int a = 0; a < outputs.size(); a++){
        for(int b = 0; b < outputs[a].size(); b++){
            cout<<"in:"<<outputs[a][b][0]<<" "<<outputs[a][b][1]<<" "<<outputs[a][b][2]<<endl;
            cout<<"out:"<<a<<endl;
        }
    }
}