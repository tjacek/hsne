#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <random>
#include <set>
#include <map>
using namespace std;

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
std::uniform_real_distribution<> dis(0.0, 1.0);


class MarkovChain{
  public:	
	MarkovChain( vector<vector<int>> states,vector<vector<double>> trans);
//	~MarkovChain();
    vector<vector<int> > states;
    vector<vector<double> > trans;
    int n_states;
    int n_dims;
    int next_state(int current_state);
    vector<int> find_landmarks(int beta,int theta,double threshold_factor);
    vector<map<int,int>> compute_influence(vector<int> landmarks,int beta);
};

void save_landmarks(const char* filename,vector<int> landmarks);
void save_influence(const char* filename,vector<map<int,int>> influence);
vector<vector<int> > to_int(vector<vector<string> > raw_strings);
vector<vector<double> > to_double(vector<vector<string> > raw_strings);
vector<vector<string> > read_file(const char* filename);
vector<string> split(const string &s, char delim);

MarkovChain::MarkovChain( vector<vector<int> > states,vector<vector<double>> trans){
  this->states=states;
  this->trans=trans;
  this->n_states= states.size();
  this->n_dims= states[0].size();
};

vector<int> MarkovChain::find_landmarks(int beta,int theta,double threshold_factor){
 	int histogram[this->n_states];
 	for(int i=0;i<this->n_states;i++){
      histogram[i]=0; 		
 	}

 	for(int state=0;state<this->n_states;state++){
      if( (state % 500) == 0){
        cout << state << endl;
      }
      for(int i=0;i<beta;i++){
      	int current_state=state;
      	for(int j=0;j<theta;j++){
      	  int raw_state=this->next_state(current_state);
      	  current_state=this->states[current_state][raw_state];	
 	    }
 	    histogram[current_state]+=1;	
 	  }	
 	}

  int landmark_threshold=threshold_factor*beta;     
 	vector<int> landmarks;
  cout << landmark_threshold << " AAAA" << endl;
 	for(int i=0;i<this->n_states;i++){
      if( histogram[i] > landmark_threshold){
        cout << histogram[i] << endl;
        landmarks.push_back(i);
      }
      /*if( (i% 70)==0){
        landmarks.push_back(i);
      }	*/	
 	}
 	return landmarks;
 }

vector<map<int,int>> MarkovChain::compute_influence(vector<int> landmarks,int beta){
  vector<map<int,int>> influence;
  set<int> landmark_set(landmarks.begin(), landmarks.end());
  map<int,int> landmark_dict;
  for(int l=0;l<landmarks.size();l++){
    landmark_dict[landmarks[l]]=l;
  }
  for(int i=0; i<this->n_states;i++){
    if( (i% 500)==0){
      cout << i <<endl;
    }
    map<int,int> histogram;
    for (int j=0; j<beta; j++){
      int current_state = i;
      set<int>::iterator result;
      result=landmark_set.find(current_state);  
      while(result==landmark_set.end()){
        int raw_state=this->next_state(current_state);
        current_state=this->states[current_state][raw_state]; 
        result=landmark_set.find(current_state); 
      }
      int landmark_index=landmark_dict[current_state];
      
      if(histogram.count(landmark_index) == 1){
        histogram[landmark_index]+=1;
      }else{
        histogram[landmark_index]=1;
      }
    }
    influence.push_back(histogram);
  }
  return influence;
}

int MarkovChain::next_state(int current_state){
   double rand_real=dis(gen);
   for(int i=0;i<this->n_dims;i++){
     if(rand_real < this->trans[current_state][i]){
     	return i;
     }
   }
   return this->n_dims;
}

void save_landmarks(const char* filename,vector<int> landmarks){
  ofstream myfile;
  myfile.open(filename);
  for(int i=0;i<landmarks.size();i++){
    myfile << landmarks[i] <<"\n";
  }
  myfile.close();
}

void save_influence(const char* filename,vector<map<int,int>> influence){
  ofstream myfile;
  myfile.open(filename);
  int n_states=influence.size();
  for(int i=0;i<n_states;i++){
    map<int,int> histogram=influence[i];
    for (map<int,int>::iterator it=histogram.begin(); it!=histogram.end(); ++it){
      myfile << "(" << it->first << "," << it->second << ")";

    }
    /*for(int j=0;j<n_landmarks;j++){
      myfile << influence[i][j];
      if(j!=(n_landmarks-1)){
        myfile <<",";
      }
    }*/
    myfile <<"\n";
  }
  myfile.close();
}

vector<vector<int> > to_int(vector<vector<string> > raw_strings){
    vector<vector<int> > result;
  int n_samples=raw_strings.size();
  int dim=raw_strings[0].size();
  std::string::size_type sz;
  for (int i=0;i<n_samples;i++){
    vector<int> sample;
    for(int j=0;j<dim;j++){
//      cout << raw_strings[i][j] << endl;	
      int value_ij= std::stoi(raw_strings[i][j],&sz);
//      cout << value_ij << endl;
      sample.push_back(value_ij);

    }
    result.push_back(sample);	
  }
  return result;
}

vector<vector<double> > to_double(vector<vector<string> > raw_strings){
    vector<vector<double> > result;
  int n_samples=raw_strings.size();
  int dim=raw_strings[0].size();
  std::string::size_type sz;
  for (int i=0;i<n_samples;i++){
    vector<double> sample;
    for(int j=0;j<dim;j++){
      double value_ij= std::stod(raw_strings[i][j],&sz);
      sample.push_back(value_ij);

    }
    result.push_back(sample);	
  }
  return result;
}

vector<vector<string> > read_file(const char* filename){
  ifstream infile(filename);
  string line;
  vector<vector<string> > result;
  while (std::getline(infile, line)) {
    vector<string> splited= split(line,',');
    result.push_back(splited);
  }
  return result;
}

vector<string> split(const string &s, char delim) {
  stringstream ss(s);
  string item;
  vector<std::string> elems;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

int main () {
  vector<vector<string> > raw_trans=read_file("trans.txt");
  vector<vector<double> > trans=to_double(raw_trans);
  vector<vector<string> > raw_states=read_file("states.txt");
  vector<vector<int> >  states=to_int(raw_states);
  MarkovChain mc(states,trans);
  vector<int> landmarks=mc.find_landmarks(100,50,7.5);
  cout << "size:"<< landmarks.size() << endl;
  save_landmarks("landmarks.txt",landmarks);
  cout << "landmarks saved" << endl;
  vector<map<int,int>> influence=mc.compute_influence(landmarks,50); 
  save_influence("influence.txt",influence);
}