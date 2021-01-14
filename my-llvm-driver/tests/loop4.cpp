int main(){
    int i = 1;
    while(i == 1){
        do{
            i++;
        L1:
            for(;i>8;i++){
                i++;
            }
        }while(i<10);
        if(i<5)
            goto L1;
    }
    return 0;
}