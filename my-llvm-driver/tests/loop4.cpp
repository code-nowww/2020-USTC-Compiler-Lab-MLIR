int main(){
    int i = 1;
    while(i == 1){
        do{
            i++;
            for(int j = 0;j < 10;j++){
                j++;
            }
        L1:
            for(;i>8;i++){
                i++;
            }
        }while(i<10);
        if(i<5)
            goto L1;
        for(int k = 0;k < 6;k++)
            k++;
    }
    return 0;
}