import filter_data
import ml_input
import analysis
import left_right_foot_classification

if __name__ == '__main__':
    filter_data.main()
    ml_input.createMlData()
    analysis.main()
    left_right_foot_classification.main()
