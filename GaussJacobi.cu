#include <iostream>
#include <math.h>
#include <vector>
using namespace std;
//Gaus-Jacobi vai ter primeiro o critério de linhas tal que: alphax é a soma dos módulos dos elementos da linha dividido pelo módulo do elemento da matriz principal
//Se passar vai pro método que precisa do chute inicial que vai fazer a próxima incógnita seja [b - (soma dos produtos das incógnitas pelos elementos da linha)] / elemento da diagonal principal

__global__
void gaussJacobi(float *matrizA_d, float *vetorB_d, float *vetXAntigo_d, float *vetXNovo_d, int ladoMatriz){
    for (int i = 0; i < ladoMatriz; i++){
        float soma = 0;
        for (int j = 0; j < ladoMatriz; j++){
            if (i != j)
                soma += matrizA_d[ladoMatriz*i+j]*vetXAntigo_d[j];
        }
        vetXNovo_d[i] = (vetorB_d[i] - soma) / matrizA_d[ladoMatriz*i+i];
    }
    
}
//aqui vamos ter uma iteração do método
int main(){
    cout << "Entre com o tamanho do lado: ";
    int N;
    cin >> N;

    int Tam = N*N;
    float *matrizA_h = new float[Tam];
    float *vetorB_h = new float[N];
    float *vetXAntigo_h = new float[N];
    float *vetXNovo_h = new float[N];
    float *vetAlphas = new float[N];
    float epsilon;
    float maxIteracoes;
    float erro = 1;
    float dif = 0;
    float iteracaoAtual = 0;
    float alphaMax = -1;

    cout << "Digite os " << Tam << " elementos da matriz: \n";
    for (int i = 0; i < Tam; i++) cin >> matrizA_h[i]; 
    
    cout << "Digite os " << N << " elementos do vetor b: \n";
    for (int i = 0; i < N; i++) cin >> vetorB_h[i]; 

    cout << "Digite os " << N << " elementos do chute: \n";
    for (int i = 0; i < N; i++) cin >> vetXAntigo_h[i]; 

    cout<<"Qual é o epsilon?\n";
    cin>>epsilon;

    cout<<"Qual o máximo de iterações?\n";
    cin>>maxIteracoes;

    float *matrizA_d;
    float *vetorB_d;
    float *vetXAntigo_d;
    float *vetXNovo_d;

    cudaMalloc((void**)&matrizA_d, Tam*(sizeof(float)));
    cudaMalloc((void**)&vetorB_d, N*(sizeof(float)));
    cudaMalloc((void**)&vetXAntigo_d, N*(sizeof(float)));
    cudaMalloc((void**)&vetXNovo_d, N*(sizeof(float)));

    cudaMemcpy(matrizA_d, matrizA_h, Tam*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vetorB_d, vetorB_h, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vetXAntigo_d, vetXAntigo_h, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vetXNovo_d, vetXNovo_h, N*sizeof(float), cudaMemcpyHostToDevice);

    int tamanho_bloco = 1<<5;
    int quant_blocos = (N+tamanho_bloco-1)/tamanho_bloco;

    for (int i = 0; i < N; i++){
        float soma = 0;
        for(int j = 0; j < N; j++){
            if (i != j)
                soma += fabs(matrizA_h[N*i+j]);
        }
        float alpha = soma /fabs(matrizA_h[N*i+i]);
        if (alpha >= alphaMax)
            alphaMax = alpha;
    }
    cout<<"Alpha max: "<<alphaMax<<endl;
    if (alphaMax > 1){
        cout<<"Alpha maior que 1";
        exit(-1);
    }
    
    //laço principal é controlado pelo host
    while (erro > epsilon && iteracaoAtual < maxIteracoes){
        gaussJacobi<<<quant_blocos, tamanho_bloco>>>(matrizA_d, vetorB_d, vetXAntigo_d, vetXNovo_d, N);
        cudaMemcpy(vetXAntigo_h, vetXAntigo_d, N*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(vetXNovo_h, vetXNovo_d, N*sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i<N; i++){
            dif = vetXAntigo_h[i] - vetXNovo_h[i];
            dif = fabs(dif);
            if (dif >= erro)
                erro = dif;
        }
        cudaMemcpy(vetXAntigo_d, vetXNovo_h, N*sizeof(float), cudaMemcpyHostToDevice);
        iteracaoAtual++;
    }
    
    cout<<"Esse é o vetor de incógnitas: "<<endl;
    for(int i=0; i<N; i++){
        cout<<vetXNovo_h[i]<<endl;
    }

    delete[] matrizA_h;
    delete[] vetorB_h;
    delete[] vetXAntigo_h;
    delete[] vetXNovo_h;
    delete[] vetAlphas;
    cudaFree(matrizA_d);
    cudaFree(vetorB_d);
    cudaFree(vetXAntigo_d);
    cudaFree(vetXNovo_d);
    return 0;
}