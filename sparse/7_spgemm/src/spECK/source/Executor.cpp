#include "cuda_runtime.h"
#include "Executor.h"
#include "Multiply.h"
#include "DataLoader.h"
#include <iomanip>
#include "Config.h"
#include "Compare.h"
#include <cusparse/include/cuSparseMultiply.h>
#include "Timings.h"
#include "spECKConfig.h"
#include <sys/time.h>

template <typename ValueType>
int Executor<ValueType>::run()
{
	iterationsWarmup = Config::getInt(Config::IterationsWarmUp, 5);
	iterationsExecution = Config::getInt(Config::IterationsExecution, 10);
	DataLoader<ValueType> data(runConfig.filePath);
	auto& matrices = data.matrices;
	// std::cout << "Matrix: " << matrices.cpuA.rows << "x" << matrices.cpuA.cols << ": " << matrices.cpuA.nnz << " nonzeros\n";
	std::cout << matrices.cpuA.rows << "," << matrices.cpuA.cols << "," << matrices.cpuA.nnz << ",";

	dCSR<ValueType> dCsrHiRes, dCsrReference;
	Timings timings, warmupTimings, benchTimings;
	bool measureAll = Config::getBool(Config::TrackIndividualTimes, false);
	bool measureCompleteTimes = Config::getBool(Config::TrackCompleteTimes, true);
	auto config = spECK::spECKConfig::initialize(0);

	bool compareData = false;

	if(Config::getBool(Config::CompareResult))
	{
		unsigned cuSubdiv_nnz = 0;
		cuSPARSE::CuSparseTest<ValueType> cusparse;
		cusparse.Multiply(matrices.gpuA, matrices.gpuB, dCsrReference, cuSubdiv_nnz);

		if(!compareData)
		{
			cudaFree(dCsrReference.data);
			dCsrReference.data = nullptr;
		}
	}

	// Warmup iterations for multiplication
	// for (int i = 0; i < iterationsWarmup; ++i)
	// {
	// 	timings = Timings();
	// 	timings.measureAll = measureAll;
	// 	timings.measureCompleteTime = measureCompleteTimes;
	// 	spECK::MultiplyspECK<ValueType, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(matrices.gpuA, matrices.gpuB, dCsrHiRes, config, timings);
	// 	warmupTimings += timings;

	// 	if (dCsrHiRes.data != nullptr && dCsrHiRes.col_ids != nullptr && Config::getBool(Config::CompareResult))
	// 	{
	// 		if (!spECK::Compare(dCsrReference, dCsrHiRes, false))
	// 			printf("Error: Matrix incorrect\n");
	// 	}
	// }

	// Multiplication

	struct timeval start, end;
    
	double time_cusparse_spgemm = 0;
	for (int i = 0; i < iterationsExecution; ++i)
	{	
		gettimeofday(&start, NULL);
		timings = Timings();
		timings.measureAll = measureAll;
		timings.measureCompleteTime = measureCompleteTimes;

		spECK::MultiplyspECK<ValueType, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(matrices.gpuA, matrices.gpuB, dCsrHiRes, config, timings);
		
		benchTimings += timings;

		if (dCsrHiRes.data != nullptr && dCsrHiRes.col_ids != nullptr && Config::getBool(Config::CompareResult))
		{
			if (!spECK::Compare(dCsrReference, dCsrHiRes, false))
				printf("Error: Matrix incorrect\n");
		}
		gettimeofday(&end, NULL);
		double single_time = ((end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0);
		time_cusparse_spgemm = time_cusparse_spgemm > single_time ? time_cusparse_spgemm : single_time;
	}
	
    // double time_cusparse_spgemm = ((end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0) / iterationsExecution;
	benchTimings /= iterationsExecution;

	// std::cout << std::setw(20) << "var-SpGEMM -> NNZ: " << dCsrHiRes.nnz << std::endl;
	// std::cout << std::setw(20) << "var-SpGEMM SpGEMM: " << benchTimings.complete << " ms" << std::endl;
	std::cout << dCsrHiRes.nnz << "," << time_cusparse_spgemm << "," << benchTimings.complete << std::endl;

	return 0;
}

template int Executor<float>::run();
template int Executor<double>::run();
