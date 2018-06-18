#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <random>

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
    void find_landmarks(int beta,int theta);
};

//MarkovChain make_markov_chain(const char* states_file,const char* trans_files);
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

void MarkovChain::find_landmarks(int beta,int theta){
 	int landmarks[this->n_states];
 	for(int i=0;i<this->n_states;i++){
      landmarks[i]=0; 		
 	}

 	for(int state=0;state<this->n_states;state++){
      if( (state % 10) == 0){
        cout << state << endl;
      }
      for(int i=0;i<beta;i++){
      	int current_state=state;
      	for(int j=0;j<theta;j++){
      	  int raw_state=this->next_state(current_state);
      	  current_state=this->states[current_state][raw_state];	
 	    }
 	    landmarks[current_state]+=1;	
 	  }	
 	}

 	for(int i=0;i<this->n_states;i++){
      cout << landmarks[i] << endl; 		
 	}
 }

int MarkovChain::next_state(int current_state){
   double rand_real=dis(gen);
//   int next;
   for(int i=0;i<this->n_dims;i++){
     if(rand_real < this->trans[current_state][i]){
       //next=rand_real;
     	return i;
     }
   }
   return this->n_dims;
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
//      cout << raw_strings[i][j] << endl;	
      double value_ij= std::stod(raw_strings[i][j],&sz);
//      cout << value_ij << endl;
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
//    cout << splited.size() << endl;
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
    // elems.push_back(std::move(item)); // if C++11 (based on comment from @mchiasson)
  }
  return elems;
}

int main () {
  vector<vector<string> > raw_trans=read_file("trans.txt");
  vector<vector<double> > trans=to_double(raw_trans);
  vector<vector<string> > raw_states=read_file("states.txt");
  vector<vector<int> >  states=to_int(raw_states);
  MarkovChain mc(states,trans);
  mc.find_landmarks(100,50);
  cout << mc.next_state(0);
}