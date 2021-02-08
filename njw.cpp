#include<bits/stdc++.h>
#include"kmeans.cpp"								/* used an implementation provided by https://github.com/aditya1601/kmeans-clustering-cpp.git*/
#include<eigen3/Eigen/Eigenvalues>					/* eigen3 lib installed using sudo apt-get install libeigen3-dev */
using namespace std;

class NJW
{
private:
	vector< vector <double>> affinity;
	vector<vector<double>> points;
	vector<vector<double>> diagonal_matr;
	vector<vector<double>> laplacian;
	vector<double> eigvalues;
	vector<vector<double>> eigvectors;
	vector<pair<double,vector<double>>> eig_pairs; /* stores eigenvalue and it's correspongding eigenvector */
	int k;

public:
	NJW(vector<vector<double> >v, int K)
	{
		points = v;
		k = K;
	}
	/* calculates the affinity between data points */
	void populateAffinity()
	{
		clock_t start, end;
		start=clock();

		int n=points.size();
		affinity.resize(n);
		for(int i=0;i<n;i++)
			affinity[i].resize(n,0);

		double sigma=1;
		for(int i=0;i<points.size();i++)
		{
			for(int j=0;j<points.size();j++)
			{
				double dist=pow(points[i][0]-points[j][0],2)+pow(points[i][1]-points[j][1],2);
				double aff=exp(-dist/(2*pow(sigma,2))); 			// sigma taken 1
				affinity[i][j]=aff;
			}
			affinity[i][i]=0;
		}

		end=clock();
		double time_taken = double(end-start)/double(CLOCKS_PER_SEC);
		cout<<"Time taken to calculate affinity matrix: "<<time_taken<<" sec\n";
	}

	/* computes the diagonal matrix */
	void populateDiagonal()
	{
		clock_t start, end;
		start=clock();

		int n=points.size();
		diagonal_matr.resize(n);
		for(int i=0;i<n;i++)
			diagonal_matr[i].resize(n,0);

		for(int i=0;i<n;i++)
		{
			double sum=0;
			for(int j=0;j<n;j++)
				sum+=affinity[i][j];
			diagonal_matr[i][i]=sum;
		}

		end=clock();
		double time_taken = double(end-start)/double(CLOCKS_PER_SEC);
		cout<<"Time taken to calculate diagonal matrix: "<<time_taken<<" sec\n";
	}

	/* calculates the laplacian, L=D^(-1/2)AD^(1/2) */
	void populateLaplacian()
	{
		clock_t start, end;
		start=clock();

		int n=points.size();
		laplacian.resize(n);
		for(int i=0;i<n;i++)
			laplacian[i].resize(n,0);

		for(int i=0;i<n;i++)
		{
			for(int j=0;j<n;j++)
				laplacian[i][j]=affinity[i][j]/(sqrt(diagonal_matr[i][i] * diagonal_matr[j][j]));
		}

		end=clock();
		double time_taken = double(end-start)/double(CLOCKS_PER_SEC);
		cout<<"Time taken to calculate laplacian matrix: "<<time_taken<<" sec\n";
	}

	void printMatrices()
	{
		cout<<"Affinity matrix: \n";
		for(int i=0;i<points.size();i++)
		{
			for(int j=0;j<points.size();j++)
				cout<<affinity[i][j]<<" ";
			cout<<endl;
		}
		cout<<"Diagonal matrix: \n";
		for(int i=0;i<diagonal_matr.size();i++)
		{
			for(int j=0;j<diagonal_matr[i].size();j++)
				cout<<diagonal_matr[i][j]<<" ";
			cout<<endl;
		}
		cout<<"Laplacian matrix: \n";
		for(int i=0;i<laplacian.size();i++)
		{
			for(int j=0;j<laplacian[i].size();j++)
				cout<<laplacian[i][j]<<" ";
			cout<<endl;
		}
	}

	void printEigen()
	{
		cout<<"Eigen values are: \n";
		for(int i=0;i<eigvalues.size();i++)
			cout<<eigvalues[i]<<" ";
		cout<<endl;

		cout<<"Eigen vectors are: \n";
		for(int i=0;i<eigvectors.size();i++)
		{
			for(int j=0;j<eigvectors[i].size();j++)
				cout<<eigvectors[i][j]<<" ";
			cout<<endl;
		}

		cout<<"K largest eigen vectors are: \n";
		for(int i=0;i<k;i++)
		{
			for(int j=0;j<eig_pairs[i].second.size();j++)
				cout<<eig_pairs[i].second[j]<<" ";
			cout<<endl;
		}
	}

	/* calculates all eigen values and eigen vectors of laplacian */
	void getEigenVectors()
	{
		clock_t start, end;
		start=clock();

		int n=laplacian.size();
		// Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A;
		Eigen::MatrixXd A(5,5);
		A.resize(n, n);
		for(int i=0;i<n;i++)
		{
			for(int j=0;j<n;j++)
			{
				A(i, j) = laplacian[i][j];
			}
		}
		// Eigen::Matrix<double, n, n> A = laplacian;
		Eigen::EigenSolver<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>> s(A); // the instance s includes the eigensystem
		for(int i=0;i<n;i++)
			eigvalues.push_back(real(s.eigenvalues()(i)));
		
		// cout<<s.eigenvectors()<<endl;
		// cout<<s.eigenvectors().shape();

		for(int i=0;i<n;i++)
		{
			vector<double> temp;
			for(int j=0;j<n;j++)
				temp.push_back(real(s.eigenvectors().col(i)(j)));

			eigvectors.push_back(temp);
		}

		end=clock();
		double time_taken = double(end-start)/double(CLOCKS_PER_SEC);
		cout<<"Time taken to calculate eigenvalues and eigenvectors: "<<time_taken<<" sec\n";
	}

	/* sort eigenvectors according to eigen values */
	void sortEigen()
	{
		clock_t start, end;
		start=clock();

		eig_pairs.clear();
		for(int i=0;i<eigvalues.size();i++)
			eig_pairs.push_back({abs(eigvalues[i]),eigvectors[i]});
		sort(eig_pairs.rbegin(),eig_pairs.rend());

		end=clock();
		double time_taken = double(end-start)/double(CLOCKS_PER_SEC);
		cout<<"Time taken to sort eigenvectors: "<<time_taken<<" sec\n";
	}

	void kmeans_aux(int iters)
	{
		clock_t start, end;
		start=clock();

		vector<vector<double>> Y;
		int n=laplacian.size();
		for(int i=0;i<n;i++)
		{
			vector<double> temp;
			for(int j=0;j<k;j++)
				temp.push_back(eig_pairs[j].second[i]);
			
			double sq_sum=0;
			for(int j=0;j<temp.size();j++)
				sq_sum+=pow(temp[j],2);
			sq_sum=sqrt(sq_sum);
			for(int j=0;j<temp.size();j++)
				temp[j]/=sq_sum;

			Y.push_back(temp);
		}

		int pointId = 0;
		vector<Point> all_points;
		
		for(int i=0;i<Y.size();i++)
		{
			Point point(pointId, Y[i]);
			all_points.push_back(point);
			pointId++;
	    }

	    KMeans kmeans(k, iters);
	    vector<Cluster> clusters = kmeans.run(all_points);

	    

	    ofstream myfile;
        myfile.open("output.csv");
        myfile << "x,y,c" << endl;

        for(int i=0; i<k; i++)
        {
            // cout<<"Points in cluster "<<clusters[i].getId()<<" : ";
            for(int j=0; j<clusters[i].getSize(); j++)
            {
            	for(int m=0;m<points[clusters[i].getPoint(j).getID()].size();m++)
            	{
            		myfile << points[clusters[i].getPoint(j).getID()][m] << ",";
            	}
                myfile << clusters[i].getId() << endl;
            }
        }

        // for(int i=0;i<) 
        // {
        //     myfile << it->x << "," << it->y << "," << it->cluster << endl;
        // }
        myfile.close();

        end=clock();
		double time_taken = double(end-start)/double(CLOCKS_PER_SEC);
		cout<<"Time taken to do K-means clustering: "<<time_taken<<" sec\n";
	}
};

int main(int argc, char **argv)
{
	clock_t start, end;
	start=clock();
	//Need 2 arguments (except filename) to run, else exit
    if(argc != 3){
        cout<<"Error: command-line argument count mismatch.";
        return 1;
    }

    //Fetching number of clusters
    int K = atoi(argv[2]);

    //Open file for fetching points
    string filename = argv[1];
    ifstream infile(filename.c_str());

    if(!infile.is_open()){
        cout<<"Error: Failed to open file."<<endl;
        return 1;
    }

    vector<vector<double> > points;
    //Fetching points from file
    string line;

    while(getline(infile, line))
    {
    	if(line=="") break;
    	vector<double> vec;
        stringstream is(line);
        double val;
        while(is >> val)
        {
            vec.push_back(val);
        }
        points.push_back(vec);
    }
    infile.close();
    cout<<"\nData fetched successfully!"<<endl<<endl;

	// = { {0,1},{1,2},{2,3},{38,48},{48,58} };
	NJW *njw = new NJW(points, K);
	
	njw->populateAffinity();
	njw->populateDiagonal();
	njw->populateLaplacian();
	njw->printMatrices();

	njw->getEigenVectors();
	njw->sortEigen();
	njw->printEigen();
	njw->kmeans_aux(100);

	end=clock();
	double time_taken = double(end-start)/double(CLOCKS_PER_SEC);
	cout<<"Time taken to run complete code: "<<time_taken<<" sec\n";

	return 0;
}

