#include<stdio.h>
int main(){
int n = 5;
int spaces = 2;
for(int i=1; i<=5; i+=2){
    for(int j=0; j<spaces; j++){
        putchar(' ');
    }
    for(int k=0; k<i; k++){
        putchar('*');
    }
    printf("\n");
    spaces--;
}
spaces+=2;
for(int i=3; i>=1; i-=2){
    for(int j=0; j<spaces; j++){
        putchar(' ');
    }
    for(int k=0; k<i; k++){
        putchar('*');
    }
    printf("\n");
    spaces++;
}
}