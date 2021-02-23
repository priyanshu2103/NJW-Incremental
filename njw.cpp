#include <bits/stdc++.h>
#include "kmeans.cpp"								/* used an implementation provided by https://github.com/aditya1601/kmeans-clustering-cpp.git*/
#include <eigen3/Eigen/Eigenvalues>					/* eigen3 lib installed using sudo apt-get install libeigen3-dev */
#include <eigen3/Eigen/Core>
#include "Spectra/SymEigsSolver.h"
using namespace std;
using namespace Spectra;

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
	int dim;
	double affinity_time;
	double laplacian_time;
	double diagonal_time;
	double eigen_time;
	double eigen_sort_time;
	double kmeans_time;

public:
	NJW(vector<vector<double> >v, int K, int d)
	{
		points = v;
		k = K;
		dim = d;
	}

	void setDimension(int d)
	{
		dim = d;
	}

	int getDimension()
	{
		return dim;
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

		// vector<double> mean(dim, 0);
		// vector<double> maxs(dim, -1);

		// for(int i=0;i<n;i++)
		// {
		// 	for(int l=0;l<dim;l++)
		// 	{
		// 		mean[l] += points[i][l]/n;
		// 		maxs[l] = max(maxs[l], points[i][l]);
		// 	}
		// }

		// for(int i=0;i<n;i++)
		// {
		// 	for(int l=0;l<dim;l++)
		// 	{
		// 		points[i][l] -= mean[l];
		// 		points[i][l] /= maxs[l];
		// 	}
		// }

		// vector<double >sigma(n);
		// for(int i=0;i<points.size();i++)
		// {
		// 	priority_queue<double, vector<double>, greater<double> > pq;
		// 	for(int j=0;j<points.size();j++)
		// 	{
		// 		double dist = 0;
		// 		for(int l=0;l<dim;l++)
		// 		{
		// 			dist += pow(points[i][l]-points[j][l],2);
		// 		}
		// 		affinity[i][j]=dist;
		// 		pq.push(dist);
		// 	}
		// 	affinity[i][i]=0;
		// 	double si = 0;
		// 	for(int l=0;l<k;l++)
		// 	{
		// 		si += sqrt(pq.top());
		// 		pq.pop();
		// 	}

		// 	si /= k;
		// 	sigma[i] = si;
		// }

		// for(int i=0;i<n;i++)
		// {
		// 	cout<<sigma[i]<<endl;
		// }
		// printMatrices();
		double sigma=1;
		for(int i=0;i<points.size();i++)
		{
			for(int j=0;j<points.size();j++)
			{
				double dist = 0;
				for(int l=0;l<dim;l++)
				{
					dist += pow(points[i][l]-points[j][l],2);
				}
				// affinity[i][j]=dist;
				affinity[i][j] = exp(-dist / (2*sigma*sigma));
			}
			affinity[i][i]=0;
		}

		end=clock();
		affinity_time = double(end-start)/double(CLOCKS_PER_SEC);
		// cout<<"Time taken to calculate affinity matrix: "<<time_taken<<" sec\n";
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
		diagonal_time = double(end-start)/double(CLOCKS_PER_SEC);
		// cout<<"Time taken to calculate diagonal matrix: "<<time_taken<<" sec\n";
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
		laplacian_time = double(end-start)/double(CLOCKS_PER_SEC);
		// cout<<"Time taken to calculate laplacian matrix: "<<time_taken<<" sec\n";
	}

	/* prints the affinity, diagonal and laplacian matrix into respective files */
	void printMatrices()
	{
		// cout<<"Affinity matrix: \n";
		// for(int i=0;i<points.size();i++)
		// {
		// 	for(int j=0;j<points.size();j++)
		// 		cout<<affinity[i][j]<<" ";
		// 	cout<<endl;
		// }
		ofstream outfile;
		outfile.open("affinity.txt");
		outfile<<"Affinity matrix: \n";
		for(int i=0;i<points.size();i++)
		{
			for(int j=0;j<points.size();j++)
				outfile<<affinity[i][j]<<" ";
			outfile<<endl;
		}
		outfile.close();

		// cout<<"Diagonal matrix: \n";
		// for(int i=0;i<diagonal_matr.size();i++)
		// {
		// 	for(int j=0;j<diagonal_matr[i].size();j++)
		// 		cout<<diagonal_matr[i][j]<<" ";
		// 	cout<<endl;
		// }

		outfile.open("diagonal.txt");
		outfile<<"Diagonal matrix: \n";
		for(int i=0;i<diagonal_matr.size();i++)
		{
			for(int j=0;j<diagonal_matr[i].size();j++)
				outfile<<diagonal_matr[i][j]<<" ";
			outfile<<endl;
		}
		outfile.close();

		// cout<<"Laplacian matrix: \n";
		// for(int i=0;i<laplacian.size();i++)
		// {
		// 	for(int j=0;j<laplacian[i].size();j++)
		// 		cout<<laplacian[i][j]<<" ";
		// 	cout<<endl;
		// }

		outfile.open("laplacian.txt");
		outfile<<"Laplacian matrix: \n";
		for(int i=0;i<laplacian.size();i++)
		{
			for(int j=0;j<laplacian[i].size();j++)
				outfile<<laplacian[i][j]<<" ";
			outfile<<endl;
		}
		outfile.close();
	}

	/* prints the eigenvalues and eigenvectors into respective files */
	void printEigen()
	{
		// cout<<"Eigen values are: \n";
		// for(int i=0;i<eigvalues.size();i++)
		// 	cout<<eigvalues[i]<<" ";
		// cout<<endl;

		cout<<"Eigen vectors are: \n";
		for(int i=0;i<eigvectors.size();i++)
		{
			for(int j=0;j<eigvectors[i].size();j++)
				cout<<eigvectors[i][j]<<" ";
			cout<<endl;
		}

		// cout<<"K largest eigen vectors are: \n";
		// for(int i=0;i<k;i++)
		// {
		// 	for(int j=0;j<eig_pairs[i].second.size();j++)
		// 		cout<<eig_pairs[i].second[j]<<" ";
		// 	cout<<endl;
		// }

		ofstream outfile;
		outfile.open("eigen.txt");
		// outfile<<"Eigen values are: \n";
		// for(int i=0;i<eigvalues.size();i++)
		// 	outfile<<eigvalues[i]<<" ";
		// outfile<<endl;

		outfile<<"Eigen vectors are: \n";
		for(int i=0;i<eigvectors.size();i++)
		{
			for(int j=0;j<eigvectors[i].size();j++)
				outfile<<eigvectors[i][j]<<" ";
			outfile<<endl;
		}

		// outfile<<"K largest eigen vectors are: \n";
		// for(int i=0;i<k;i++)
		// {
		// 	for(int j=0;j<eig_pairs[i].second.size();j++)
		// 		outfile<<eig_pairs[i].second[j]<<" ";
		// 	outfile<<endl;
		// }
		outfile.close();
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

		DenseSymMatProd<double> op(A);

    // Construct eigen solver object, requesting the largest three eigenvalues
    SymEigsSolver<DenseSymMatProd<double>> eigs(op, k, 3*k); // 1 is arbitrary, k is for k evectors

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute(SortRule::LargestAlge);

    // Retrieve results
    Eigen::MatrixXd evectors;
    if(eigs.info() == CompInfo::Successful)
        evectors = eigs.eigenvectors();

    cout << "Eigenvectors found:\n" << evectors << endl;

		eigvectors.resize(n, vector<double> (k));
		for(int i=0;i<n;i++)
		{
			double sq_sum = 0;
			for(int j=0;j<k;j++)
			{
				eigvectors[i][j] = evectors(i, j);
				sq_sum += pow(evectors(i, j), 2);
			}
			for(int j=0;j<k;j++)
			{
				eigvectors[i][j] /= sq_sum;
			}
		}

		// eig_pairs.clear();
		// for(int i=0;i<eigvalues.size();i++)
		// 	eig_pairs.push_back({abs(eigvalues[i]),eigvectors[i]});
		Eigen::VectorXd evalues;
    if(eigs.info() == CompInfo::Successful)
        evalues = eigs.eigenvalues();

    cout << "Eigenvalues found:\n" << evalues << endl;

		// eigvectors = vector<vector<double> > (evectors);
		// Eigen::Matrix<double, n, n> A = laplacian;
		// Eigen::EigenSolver<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>> s(A); // the instance s includes the eigensystem
		// for(int i=0;i<n;i++)
		// 	eigvalues.push_back(real(s.eigenvalues()(i)));
		//
		// // cout<<s.eigenvectors()<<endl;
		// // cout<<s.eigenvectors().shape();
		//
		// for(int i=0;i<n;i++)
		// {
		// 	vector<double> temp;
		// 	for(int j=0;j<n;j++)
		// 		temp.push_back(real(s.eigenvectors().col(i)(j)));
		//
		// 	eigvectors.push_back(temp);
		// }

		end=clock();
		eigen_time = double(end-start)/double(CLOCKS_PER_SEC);
		// cout<<"Time taken to calculate eigenvalues and eigenvectors: "<<time_taken<<" sec\n";
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
		eigen_sort_time = double(end-start)/double(CLOCKS_PER_SEC);
		// cout<<"Time taken to sort eigenvectors: "<<time_taken<<" sec\n";
	}

	/* applies the k-means algorithm on the normalized matrix of eigenvectors */
	void kmeans_aux(int iters)
	{
		clock_t start, end;
		start=clock();

		// vector<vector<double>> Y;
		// int n=laplacian.size();
		// for(int i=0;i<n;i++)
		// {
		// 	vector<double> temp;
		// 	for(int j=0;j<k;j++)
		// 		temp.push_back(eig_pairs[j].second[i]);
		//
		// 	double sq_sum=0;
		// 	for(int j=0;j<temp.size();j++)
		// 		sq_sum+=pow(temp[j],2);
		// 	sq_sum=sqrt(sq_sum);
		// 	for(int j=0;j<temp.size();j++)
		// 		temp[j]/=sq_sum;
		//
		// 	Y.push_back(temp);
		// }

		int pointId = 0;
		vector<Point> all_points;

		for(int i=0;i<eigvectors.size();i++)
		{
			Point point(pointId, eigvectors[i]);
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
		kmeans_time = double(end-start)/double(CLOCKS_PER_SEC);
		// cout<<"Time taken to do K-means clustering: "<<time_taken<<" sec\n";
	}

	/* prints the running time for all the steps of the algorithm */
	void printFuncTimes(double total_time)
	{
		cout<<"Time taken to calculate affinity matrix: "<<affinity_time<<" sec\n";
		cout<<"Time taken to calculate diagonal matrix: "<<diagonal_time<<" sec\n";
		cout<<"Time taken to calculate laplacian matrix: "<<laplacian_time<<" sec\n";
		cout<<"Time taken to calculate eigenvalues and eigenvectors: "<<eigen_time<<" sec\n";
		cout<<"Time taken to sort eigenvectors: "<<eigen_sort_time<<" sec\n";
		cout<<"Time taken to do K-means clustering: "<<kmeans_time<<" sec\n";
		cout<<"Time taken to run complete code: "<<total_time<<" sec\n";

		ofstream outfile;
		outfile.open("timer.txt");
		outfile<<"Time taken to calculate affinity matrix: "<<affinity_time<<" sec\n";
		outfile<<"Time taken to calculate diagonal matrix: "<<diagonal_time<<" sec\n";
		outfile<<"Time taken to calculate laplacian matrix: "<<laplacian_time<<" sec\n";
		outfile<<"Time taken to calculate eigenvalues and eigenvectors: "<<eigen_time<<" sec\n";
		outfile<<"Time taken to sort eigenvectors: "<<eigen_sort_time<<" sec\n";
		outfile<<"Time taken to do K-means clustering: "<<kmeans_time<<" sec\n";
		outfile<<"Time taken to run complete code: "<<total_time<<" sec\n";

		outfile.close();
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

    int d;
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
        d = vec.size();
    }
    infile.close();
    cout<<"\nData fetched successfully!"<<endl<<endl;

	// = { {0,1},{1,2},{2,3},{38,48},{48,58} };
	NJW *njw = new NJW(points, K, d);

	njw->populateAffinity();
	njw->populateDiagonal();
	njw->populateLaplacian();
	njw->printMatrices();

	njw->getEigenVectors();
	// njw->sortEigen();
	njw->printEigen();
	njw->kmeans_aux(100);

	end=clock();
	double time_taken = double(end-start)/double(CLOCKS_PER_SEC);

	njw->printFuncTimes(time_taken);

	return 0;
}
