#include<bits/stdc++.h>
#include <eigen3/Eigen/Eigenvalues>
using namespace std;

vector< vector <double>> affinity;
vector<vector<int>> points;
vector<vector<double>> diagonal_matr;
vector<vector<double>> laplacian;
vector<double> eigvalues;
vector<vector<double>> eigvectors;
vector<pair<double,vector<double>>> eig_pairs;
int k=2;


void populateAffinity()
{
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
}

void populateDiagonal()
{
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

void populateLaplacian()
{
	int n=points.size();
	laplacian.resize(n);
	for(int i=0;i<n;i++)
		laplacian[i].resize(n,0);

	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
			laplacian[i][j]=affinity[i][j]/(sqrt(diagonal_matr[i][i] * diagonal_matr[j][j]));
	}
}

void getEigenVectors()
{
	const int n=laplacian.size();
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
}

void extractKeigen()
{
	eig_pairs.clear();
	for(int i=0;i<eigvalues.size();i++)
		eig_pairs.push_back({abs(eigvalues[i]),eigvectors[i]});
	sort(eig_pairs.rbegin(),eig_pairs.rend());
}

int main()
{
	points= { {0,1},{1,2},{2,3},{3,4},{4,5} };
	populateAffinity();
	populateDiagonal();
	populateLaplacian();
	printMatrices();

	getEigenVectors();
	extractKeigen();
	printEigen();

	return 0;
}

