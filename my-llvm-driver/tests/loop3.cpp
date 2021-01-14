void funLoop(){
    for(int i =0;i < 20;i++){
        i++;
        for(int j = 0;j < 10;j++)
            j++;
    }
}

int main(){
    for(int i=0;i < 10;i++){
        i++;
        funLoop();
    }
    return 0;
}
