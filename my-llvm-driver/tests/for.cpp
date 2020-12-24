int main(){
    L1:
    for(int i=0;i<10;i++){
        i++;
        for(int i=0;i<10;i++){
            i++;
            for(int i=0;i<10;i++){
                i++;
            }
            for(int i=0;i<10;i++){
                i++;
            }
        }
    }
    for(int i=0;i<10;i++){
        i++;
    }
    int j = 0;
    goto L1;
    return 0;
}