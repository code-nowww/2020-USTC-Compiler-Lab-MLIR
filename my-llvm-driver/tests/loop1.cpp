int main(){
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 10; j++){
            if(i < j){
                break;
            }
            else{
                continue;
            }
        }
    }
}