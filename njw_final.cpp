#include<bits/stdc++.h>
#include"kmeans.cpp"								/* used an implementation provided by https://github.com/aditya1601/kmeans-clustering-cpp.git*/
#include<eigen3/Eigen/Eigenvalues>					/* eigen3 lib installed using sudo apt-get install libeigen3-dev */
using namespace std;


class NJW
{
private:
	vector<vector<double>> affinity;
	vector<vector<double>> points;
	vector<vector<double>> diagonal_matr;
	vector<vector<double>> laplacian;
	vector<double> eigvalues;
	vector<vector<double>> eigvectors;
	vector<pair<double,vector<double>>> eig_pairs; /* stores eigenvalue and it's correspongding eigenvector */
	int k;
	int dim;
	int iters;
	string infile;
	string type;
	double sigma;
	double affinity_time;
	double laplacian_time;
	double diagonal_time;
	double eigen_time;
	double eigen_sort_time;
	double kmeans_time;
	map<int,int> IdToCluster;

public:
	NJW(vector<vector<double> >v, int K, int d, double s, int its, string file)
	{
		points = v;
		k = K;
		dim = d;
		sigma = s;
		iters = its;
		infile = file;
		type = "static_orig";
	}

	void setIters(int it)
	{
		iters = it;
	}

	int getIters()
	{
		return iters;
	}

	void setDimension(int d)
	{
		dim = d;
	}

	int getDimension()
	{
		return dim;
	}

	int getNumPoints()
	{
		return points.size();
	}

	void setSigma(double s)
	{
		sigma = s;
	}

	int getSigma()
	{
		return sigma;
	}

	void setType(string s)
	{
		type = s;
	}

	string getType()
	{
		return type;
	}

	vector<vector<double> > getAffinity()
	{
		return affinity;
	}

	vector<vector<double> > getDiagonal()
	{
		return diagonal_matr;
	}

	vector<vector<double> > getLaplacian()
	{
		return laplacian;
	}

	vector<vector<double> > getPoints()
	{
		return points;
	}

	vector<vector<double> > getEigVectors()
	{
		return eigvectors;
	}

	vector<double> getEigValues()
	{
		return eigvalues;
	}

	int getK()
	{
		return k;
	}

	void setAffinity(vector<vector<double> > v)
	{
		affinity = v;
	}

	void setDiagonal(vector<vector<double> > v)
	{
		diagonal_matr = v;
	}

	void setLaplacian(vector<vector<double> > v)
	{
		laplacian = v;
	}

	void setPoints(vector<vector<double> > v)
	{
		points = v;
	}

	void setEigVectors(vector<vector<double> > v)
	{
		eigvectors = v;
	}

	void setEigValues(vector<double> v)
	{
		eigvalues = v;
	}

	void setK(int new_k)
	{
		k = new_k;
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
		outfile.open(type + "_" + infile + "_affinity.txt");
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

		outfile.open(type + "_" + infile + "_diagonal.txt");
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

		outfile.open(type + "_" + infile + "_laplacian.txt");
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
		cout<<"Eigen values are: \n";
		for(int i=0;i<eigvalues.size();i++)
			cout<<eigvalues[i]<<" ";
		cout<<endl;

		// cout<<"Eigen vectors are: \n";
		// for(int i=0;i<eigvectors.size();i++)
		// {
		// 	for(int j=0;j<eigvectors[i].size();j++)
		// 		cout<<eigvectors[i][j]<<" ";
		// 	cout<<endl;
		// }

		cout<<"K largest eigen vectors are: \n";
		for(int i=0;i<k;i++)
		{
			for(int j=0;j<eig_pairs[i].second.size();j++)
				cout<<eig_pairs[i].second[j]<<" ";
			cout<<endl;
		}

		ofstream outfile;
		outfile.open(type + "_" + infile + "_eigen.txt");
		outfile<<"Eigen values are: \n";
		for(int i=0;i<eigvalues.size();i++)
			outfile<<eigvalues[i]<<" ";
		outfile<<endl;

		outfile<<"Eigen vectors are: \n";
		for(int i=0;i<eigvectors.size();i++)
		{
			for(int j=0;j<eigvectors[i].size();j++)
				outfile<<eigvectors[i][j]<<" ";
			outfile<<endl;
		}

		outfile<<"K largest eigen vectors are: \n";
		for(int i=0;i<k;i++)
		{
			for(int j=0;j<eig_pairs[i].second.size();j++)
				outfile<<eig_pairs[i].second[j]<<" ";
			outfile<<endl;
		}
		outfile.close();
	}

	/* calculates all eigen values and eigen vectors of laplacian */
	void getEigenVectors()
	{
		clock_t start, end;
		start=clock();
		eigvalues.clear();
		eigvectors.clear();

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
			double zs = 0;
			for(int j=0;j<n;j++)
			{
				double z =real(s.eigenvectors().col(i)(j));
				temp.push_back(z);
				zs += z*z;
			}

			zs = sqrt(zs);
			for(int j=0;j<n;j++)
			{
				temp[j] /= zs;
			}


			eigvectors.push_back(temp);
		}

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
			eig_pairs.push_back({eigvalues[i], eigvectors[i]});
		sort(eig_pairs.rbegin(),eig_pairs.rend());

		end=clock();
		eigen_sort_time = double(end-start)/double(CLOCKS_PER_SEC);
		// cout<<"Time taken to sort eigenvectors: "<<time_taken<<" sec\n";
	}

	/* applies the k-means algorithm on the normalized matrix of eigenvectors */
	void kmeans_aux()
	{
		string outfile = type + "_" + infile + "_output.csv";
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
	    myfile.open(outfile);
      myfile << "x,y,c"<<endl;
			IdToCluster.clear();

	    for(int i=0;i<k;i++)
	    {
	    	for(int j=0; j<clusters[i].getSize(); j++)
        {
        	// for(int m=0;m<points[clusters[i].getPoint(j).getID()].size();m++)
        	// {
        		// myfile << points[clusters[i].getPoint(j).getID()][m] << ",";
        		IdToCluster[clusters[i].getPoint(j).getID()]=clusters[i].getId();
        		// cout<<clusters[i].getPoint(j).getID()<<" "<<clusters[i].getId()<<endl;
        	// }
            // myfile << clusters[i].getId() << endl;
        }
	    }

	    for(auto it=IdToCluster.begin();it!=IdToCluster.end();it++)
	    {
	    	int id=it->first;
	    	for(int m=0;m<points[id].size();m++)
	    	{
	    		myfile << points[id][m] << ",";
	    	}
	    	myfile << it->second << endl;
	    	// cout<<it->first<<" "<<it->second<<endl;
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
		outfile.open(type + "_" + infile + "_timer.txt");
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


class Incremental
{
private:
	NJW *njw;
	int order;

public:
	Incremental(NJW *nj, int ord)
	{
		njw = nj;
		order = ord;
	}

	double matrMult(vector<double> x1, vector<vector<double> > L, vector<double> x2) // (x1T)L(x2)
	{
		vector<double> temp;
		for(int i=0;i<L.size();i++)
		{
			double s = 0;
			for(int j=0;j<L[i].size();j++)
			{
				s += L[i][j] * x2[j];
			}
			temp.push_back(s);
		}

		double res = 0;
		for(int i=0;i<x1.size();i++)
		{
			res += x1[i] * temp[i];
		}

		return res;
	}

	vector<double> addVectors(vector<double> v1, vector<double> v2)
	{
		vector<double> sum(v1.size());
		for(int i=0;i<v1.size();i++)
		{
			sum[i] = v1[i] + v2[i];
		}

		return sum;
	}

	vector<double> scalarMult(vector<double> v, double f)
	{
		vector<double> res(v.size());
		for(int i=0;i<v.size();i++)
		{
			res[i] = f*v[i];
		}
		return res;
	}

	double dot_prod(vector<double> x1, vector<double> x2)
	{
		double sum = 0;
		for(int i=0;i<x1.size();i++)
		{
			sum += x1[i] * x2[i];
		}
		return sum;
	}

	void updateEigs(vector<double> &old_eigvalues, vector<vector<double> > &old_eigvectors, vector<vector<double> > delta_l)
	{
		if(order == 1)
		{
			int N = old_eigvalues.size(); // N = n+1 for insert, n for update and remove
			vector<double> new_eigvalues(N);
			vector<vector<double> > new_eigvectors(N, vector<double> (N));

			for(int i=0;i<N;i++)
			{
				new_eigvalues[i] = old_eigvalues[i] + matrMult(old_eigvectors[i], delta_l, old_eigvectors[i]);
			}

			for(int i=0;i<N;i++)
			{
				new_eigvectors[i] = old_eigvectors[i];
				vector<double> temp_vec(N);
				for(int j=0;j<N;j++)
				{
					if(j == i) continue;
					new_eigvectors[i] = addVectors(new_eigvectors[i], scalarMult(old_eigvectors[j], (matrMult(old_eigvectors[j], delta_l, old_eigvectors[i])/(old_eigvalues[i] - old_eigvalues[j]))));
				}
			}

			for(int i=0;i<N;i++)
			{
				double sum = 0;
				for(int j=0;j<N;j++)
				{
					sum += new_eigvectors[i][j] * new_eigvectors[i][j];
				}
				sum = sqrt(sum);
				for(int j=0;j<N;j++)
				{
					new_eigvectors[i][j] /= sum;
				}
			}
			// since passed by reference
			old_eigvalues = new_eigvalues;
			old_eigvectors = new_eigvectors;
		}

		else
		{
			int N = old_eigvalues.size(); // N = n+1 for insert, n for update and remove
			vector<double> new_eigvalues(N);
			vector<vector<double> > new_eigvectors(N, vector<double> (N));

			vector<double> delta_lambda(N, 0);
			vector<double> delta_2_lambda(N, 0);

			vector<vector<double> > delta_x(N, vector<double> (N, 0));
			vector<vector<double> > delta_2_x(N, vector<double> (N, 0));

			double start = clock();
			for(int i=0;i<N;i++)
			{
				// delta_x[i] = old_eigvectors[i];
				vector<double> temp_vec(N);
				for(int j=0;j<N;j++)
				{
					if(j == i) continue;
					delta_x[i] = addVectors(delta_x[i], scalarMult(old_eigvectors[j], (matrMult(old_eigvectors[j], delta_l, old_eigvectors[i])/(old_eigvalues[i] - old_eigvalues[j]))));
				}
			}
			double end = clock();
			double time_taken = double(end-start)/double(CLOCKS_PER_SEC);
			cout<<"for loop time "<<time_taken<<endl;
			for(int i=0;i<N;i++)
			{
				delta_lambda[i] = matrMult(old_eigvectors[i], delta_l, old_eigvectors[i]);
			}

			for(int i=0;i<N;i++)
			{
				delta_2_lambda[i] = matrMult(old_eigvectors[i], delta_l, delta_x[i]) - delta_lambda[i] * dot_prod(old_eigvectors[i], delta_x[i]);
			}

			for(int i=0;i<N;i++)
			{
				// delta_x[i] = old_eigvectors[i];
				vector<double> temp_vec(N);
				for(int j=0;j<N;j++)
				{
					if(j == i)
					{
						delta_2_x[i] = addVectors(delta_2_x[i], scalarMult(old_eigvectors[j], -0.5 * dot_prod(delta_x[i], delta_x[i])));
					}
					else
					{
						delta_2_x[i] = addVectors(delta_2_x[i], scalarMult(old_eigvectors[j], (matrMult(old_eigvectors[j], delta_l, delta_x[i]) - delta_lambda[i] * dot_prod(old_eigvectors[j], delta_x[i]))/(old_eigvalues[i] - old_eigvalues[j])));
					}
				}
			}

			for(int i=0;i<N;i++)
			{
				new_eigvalues[i] = old_eigvalues[i] + delta_lambda[i] + delta_2_lambda[i];
				new_eigvectors[i] = addVectors(old_eigvectors[i], addVectors(delta_x[i], delta_2_x[i]));

				double sum = 0;
				for(int j=0;j<N;j++)
				{
					sum += new_eigvectors[i][j] * new_eigvectors[i][j];
				}
				sum = sqrt(sum);
				for(int j=0;j<N;j++)
				{
					new_eigvectors[i][j] /= sum;
				}
			}

			// since passed by reference
			old_eigvalues = new_eigvalues;
			old_eigvectors = new_eigvectors;
		}

	}

	int getClosestToZero(vector<double> eigvalues)
	{
		int index = 0;
		double minDist;
		minDist = abs(eigvalues[0]);
		for(int i=1;i<eigvalues.size();i++)
		{
			if(abs(eigvalues[i]) < minDist)
			{
				minDist = abs(eigvalues[i]);
				index = i;
			}
		}
		return index;
	}

	void update(int index, vector<double> pt)
	{
		cout<<"Updating point "<<index<<endl;
		int n = njw->getNumPoints();
		vector<vector<double> > lapl_old = njw->getLaplacian();
		vector<vector<double> > points = njw->getPoints();
		points[index] = pt;
		njw->setPoints(points);
		njw->populateAffinity();
		njw->populateDiagonal();
		njw->populateLaplacian();
		njw->printMatrices();
		vector<vector<double> > delta_l = njw->getLaplacian();
		for(int i=0;i<n;i++)
		{
			for(int j=0;j<n;j++)
			{
				delta_l[i][j] -= lapl_old[i][j];
			}
		}

		vector<double> eigvalues = njw->getEigValues();
		vector<vector<double> > eigvectors = njw->getEigVectors();

		updateEigs(eigvalues, eigvectors, delta_l);

		njw->setEigValues(eigvalues);
		njw->setEigVectors(eigvectors);

		njw->sortEigen();
		njw->printEigen();
		njw->kmeans_aux();
	}

	void insert(vector<double> pt)
	{
		cout<<"Inserting point"<<endl;
		int n = njw->getNumPoints();
		vector<vector<double> > points = njw->getPoints();
		points.push_back(pt); // This changes njw->getNumPoints()
		njw->setPoints(points);

		vector<double> dists;
		double sigma = njw->getSigma();
		double dist_sum = 0;
		for(int i=0;i<n;i++)
		{
			double dist = 0;
			for(int l=0;l<njw->getDimension();l++)
			{
				dist += pow(points[i][l]-pt[l],2);
			}
			double temp = exp(-dist / (2*sigma*sigma));
			dists.push_back(temp);
			dist_sum += temp;
		}

		vector<vector<double> > aff = njw->getAffinity();
		for(int i=0;i<aff.size();i++)
		{
			aff[i].push_back(dists[i]);
		}
		aff.push_back(dists);
		aff[aff.size()-1].push_back(0);

		njw->setAffinity(aff);

		vector<vector<double> > diag = njw->getDiagonal();
		for(int i=0;i<n;i++)
		{
			diag[i][i] += dists[i];
			diag[i].push_back(0);
		}

		vector<double> v_temp(n+1, 0);
		v_temp[n] = dist_sum;
		diag.push_back(v_temp);

		njw->setDiagonal(diag);

		vector<vector<double> > lapl_old = njw->getLaplacian();

		vector<vector<double> > lapl(n+1, vector<double>(n+1));
		for(int i=0;i<n+1;i++)
		{
			for(int j=0;j<n+1;j++)
			{
				lapl[i][j] = aff[i][j]/(sqrt(diag[i][i] * diag[j][j]));
			}
		}

		njw->setLaplacian(lapl);
		njw->printMatrices();

		vector<vector<double> > delta_l = lapl;
		for(int i=0;i<n;i++)
		{
			for(int j=0;j<n;j++)
			{
				delta_l[i][j] = lapl[i][j] - lapl_old[i][j];
			}
		}

		Eigen::MatrixXd A(5,5);
		A.resize(n+1, n+1);
		for(int i=0;i<n;i++)
		{
			for(int j=0;j<n;j++)
			{
				A(i, j) = lapl_old[i][j];
			}
		}

		for(int i=0;i<n;i++)
		{
			A(i, n) = 0;
			A(n, i) = 0;
		}

		A(n, n) = 0;

		Eigen::MatrixXd x = A.fullPivLu().kernel();

		vector<double> eigvalues = njw->getEigValues();
		vector<vector<double> > eigvectors = njw->getEigVectors();

		eigvalues.push_back(0);
		for(int i=0;i<n;i++)
		{
			eigvectors[i].push_back(0);
		}
		vector<double> temp_eig(n+1);
		for(int i=0;i<n+1;i++)
		{
			temp_eig[i] = x(i, 0);
		}

		eigvectors.push_back(temp_eig);
		updateEigs(eigvalues, eigvectors, delta_l);

		njw->setEigValues(eigvalues);
		njw->setEigVectors(eigvectors);

		njw->sortEigen();
		njw->printEigen();
		njw->kmeans_aux();
	}

	void remove(int index)
	{
		cout<<"Deleting point "<<index<<endl;
		int n = njw->getNumPoints();
		vector<vector<double> > lapl_old = njw->getLaplacian();
		vector<vector<double> > points = njw->getPoints();
		points.erase(points.begin() + index);
		njw->setPoints(points);
		njw->populateAffinity();
		njw->populateDiagonal();
		njw->populateLaplacian();
		njw->printMatrices();
		vector<vector<double> > delta_l = njw->getLaplacian();
		for(int i=0;i<n-1;i++)
		{
			delta_l[i].insert(delta_l[i].begin() + index, 0);
		}
		vector<double> temp(n, 0);
		delta_l.insert(delta_l.begin() + index, temp);

		for(int i=0;i<n;i++)
		{
			for(int j=0;j<n;j++)
			{
				delta_l[i][j] -= lapl_old[i][j];
			}
		}
		vector<double> eigvalues = njw->getEigValues();
		vector<vector<double> > eigvectors = njw->getEigVectors();

		updateEigs(eigvalues, eigvectors, delta_l);

		int zero_id = getClosestToZero(eigvalues);
		eigvectors.erase(eigvectors.begin() + zero_id);
		eigvalues.erase(eigvalues.begin() + zero_id);

		for(int i=0;i<n-1;i++)
		{
			eigvectors[i].erase(eigvectors[i].begin() + index);
		}

		njw->setEigValues(eigvalues);
		njw->setEigVectors(eigvectors);

		njw->sortEigen();
		njw->printEigen();
		njw->kmeans_aux();
	}

};

int main(int argc, char **argv)
{
		clock_t start, end;
		start=clock();

    //Fetching number of clusters
    int K = stoi(argv[2]);

    //Open file for fetching points
    string filename = argv[1];
    ifstream infile(filename.c_str());

    if(!infile.is_open())
		{
        cout<<"Error: Failed to open file."<<endl;
        return 1;
    }

    vector<vector<double> > points;
		vector<double> maxs, mins;
    //Fetching points from file
    string line;

    int d;
    while(getline(infile, line))
    {
    	if(line=="") break;
    	vector<double> vec;
      stringstream is(line);
      double val;
			int idx = 0;
      while(is >> val)
      {
				if(maxs.size()<=idx)
				{
					maxs.push_back(val);
					mins.push_back(val);
				}
				else
				{
					maxs[idx] = max(maxs[idx], val);
					mins[idx] = min(mins[idx], val);
				}

				idx++;
				vec.push_back(val);
      }
      points.push_back(vec);
      d = vec.size();
    }
    infile.close();
		// int p = 1; // points.size() / 10;
		// vector<double> temp_added(d);
		// for(int i=0;i<d;i++)
		// {
		// 	temp_added[i] = 1.25*maxs[i]; //+ (maxs[i] - mins[i]);
		// 	cout<<maxs[i]<<" "<<mins[i]<<" "<<temp_added[i]<<endl;
		// }
		//
		// for(int i=0;i<p;i++)
		// {
		// 	points.push_back(temp_added);
		// }

    cout<<"\nData fetched successfully!"<<endl<<endl;

		int ord = 1; //stoi(argv[3]);
		string oper = argv[4];

		// = { {0,1},{1,2},{2,3},{38,48},{48,58} };
		NJW *njw = new NJW(points, K , d, 1, 100, filename);

		njw->populateAffinity();
		njw->populateDiagonal();
		njw->populateLaplacian();
		njw->printMatrices();

		njw->getEigenVectors();
		njw->sortEigen();
		njw->printEigen();
		njw->kmeans_aux();

		end=clock();
		double time_taken = double(end-start)/double(CLOCKS_PER_SEC);

		njw->printFuncTimes(time_taken);

		Incremental *inc = new Incremental(njw, ord);
		vector<double> pt_new;

		start = clock();
		if(oper == "update")
		{
			njw->setType("update_" + to_string(ord));
			int idx = stoi(argv[5]);
			for(int i=0;i<d;i++)
			{
				pt_new.push_back(stod(argv[6+i]));
			}
			inc -> update(idx, pt_new);
		}

		else if(oper == "insert")
		{
			njw->setType("insert_" + to_string(ord));
			for(int i=0;i<d;i++)
			{
				pt_new.push_back(stod(argv[5+i]));
			}
			// inc -> update(points.size() - 1, pt_new);
			inc -> insert(pt_new);
		}

		else
		{
			njw->setType("remove_" + to_string(ord));
			int idx = stoi(argv[5]);
			inc->remove(idx);
		}
		end = clock();
		time_taken = double(end-start)/double(CLOCKS_PER_SEC);
		njw->printFuncTimes(time_taken);

		start = clock();
		njw->setType("static_new_" + oper);
		njw->getEigenVectors();
		njw->sortEigen();
		njw->printEigen();
		njw->kmeans_aux();
		end = clock();
		time_taken = double(end-start)/double(CLOCKS_PER_SEC);
		njw->printFuncTimes(time_taken);

		return 0;
}
