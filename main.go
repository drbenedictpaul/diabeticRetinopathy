package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
)

type Patient struct {
	Features []float64
	Label    int
}

type Tree struct {
	Feature    int
	Threshold  float64
	Left       *Tree
	Right      *Tree
	Prediction int
}

type RandomForest struct {
	Trees []Tree
}

func main() {
	file, err := os.Open("diabeticRetinopathy.csv")
	if err != nil {
		fmt.Println("Error opening file:", err)
		return
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Println("Error reading CSV:", err)
		return
	}

	data := records[1:]
	patients := make([]Patient, 0, len(data))

	for _, row := range data {
		label := mapLabel(row[0])
		features := make([]float64, 0, len(row)-1)
		for i := 1; i < len(row); i++ {
			if row[i] == "NaN" || row[i] == "" {
				features = append(features, 0.0)
			} else if i == 4 {
				if row[i] == "M" {
					features = append(features, 0.0)
				} else {
					features = append(features, 1.0)
				}
			} else if i == 7 {
				features = append(features, float64(mapAlbuminuria(row[i])))
			} else {
				val, _ := strconv.ParseFloat(strings.TrimSpace(row[i]), 64)
				features = append(features, val)
			}
		}
		patients = append(patients, Patient{Features: features, Label: label})
	}

	fmt.Println("Loaded", len(patients), "patients")
	imputeMissing(patients)
	fmt.Println("Imputed missing values")
	scaleFeatures(patients)
	fmt.Println("Scaled features")

	folds := kFoldSplit(patients, 5)
	avgAccuracy := 0.0
	for i := 0; i < 5; i++ {
		test := folds[i]
		train := []Patient{}
		for j := 0; j < 5; j++ {
			if j != i {
				train = append(train, folds[j]...)
			}
		}
		fmt.Printf("Fold %d: Train %d, Test %d\n", i+1, len(train), len(test))
		forest := trainRandomForest(train, 10, 5, 1)
		fmt.Println("Random Forest trained")
		evaluateModel(forest, test)
		correct := 0
		for _, p := range test {
			if predictRandomForest(forest, p) == p.Label {
				correct++
			}
		}
		foldAccuracy := float64(correct) / float64(len(test)) * 100
		avgAccuracy += foldAccuracy
	}
	avgAccuracy /= 5
	fmt.Printf("Average Cross-Validation Accuracy: %.2f%%\n", avgAccuracy)
}

func mapLabel(category string) int {
	switch category {
	case "DM":
		return 0
	case "DR":
		return 1
	case "DN":
		return 2
	default:
		return -1
	}
}

func mapAlbuminuria(value string) int {
	switch value {
	case "Neg":
		return 0
	case "1+":
		return 1
	case "2+":
		return 2
	case "3+":
		return 3
	case "4+":
		return 4
	default:
		return 0
	}
}

func imputeMissing(patients []Patient) {
	featureCount := len(patients[0].Features)
	sums := [3][]float64{make([]float64, featureCount), make([]float64, featureCount), make([]float64, featureCount)}
	counts := [3][]int{make([]int, featureCount), make([]int, featureCount), make([]int, featureCount)}

	for _, p := range patients {
		label := p.Label
		for i, val := range p.Features {
			if val != 0.0 || (i == 4 || i == 7) {
				sums[label][i] += val
				counts[label][i]++
			}
		}
	}

	means := [3][]float64{make([]float64, featureCount), make([]float64, featureCount), make([]float64, featureCount)}
	for label := 0; label < 3; label++ {
		for i := range means[label] {
			if counts[label][i] > 0 {
				means[label][i] = sums[label][i] / float64(counts[label][i])
			}
		}
	}

	for i := range patients {
		label := patients[i].Label
		for j := range patients[i].Features {
			if patients[i].Features[j] == 0.0 && j != 4 && j != 7 {
				patients[i].Features[j] = means[label][j]
			}
		}
	}
}

func scaleFeatures(patients []Patient) {
	featureCount := len(patients[0].Features)
	means := [3][]float64{make([]float64, featureCount), make([]float64, featureCount), make([]float64, featureCount)}
	variances := [3][]float64{make([]float64, featureCount), make([]float64, featureCount), make([]float64, featureCount)}
	counts := [3]int{0, 0, 0}

	for _, p := range patients {
		label := p.Label
		counts[label]++
		for i, val := range p.Features {
			means[label][i] += val
		}
	}
	for label := 0; label < 3; label++ {
		for i := range means[label] {
			means[label][i] /= float64(counts[label])
		}
	}

	for _, p := range patients {
		label := p.Label
		for i, val := range p.Features {
			diff := val - means[label][i]
			variances[label][i] += diff * diff
		}
	}
	for label := 0; label < 3; label++ {
		for i := range variances[label] {
			variances[label][i] /= float64(counts[label])
		}
	}

	for i := range patients {
		label := patients[i].Label
		for j := range patients[i].Features {
			if j != 4 && j != 7 {
				stdDev := math.Sqrt(variances[label][j])
				if stdDev > 0 {
					patients[i].Features[j] = (patients[i].Features[j] - means[label][j]) / stdDev
				}
			}
		}
	}
}

func kFoldSplit(patients []Patient, k int) [][]Patient {
	rand.Seed(42) // Fixed seed for reproducibility
	byClass := [3][]Patient{{}, {}, {}}
	for _, p := range patients {
		byClass[p.Label] = append(byClass[p.Label], p)
	}

	for i := 0; i < 3; i++ {
		rand.Shuffle(len(byClass[i]), func(j, k int) {
			byClass[i][j], byClass[i][k] = byClass[i][k], byClass[i][j]
		})
	}

	folds := make([][]Patient, k)
	for i := range folds {
		folds[i] = []Patient{}
	}
	for i := 0; i < 3; i++ {
		for j, p := range byClass[i] {
			folds[j%k] = append(folds[j%k], p)
		}
	}
	return folds
}

func trainTree(train []Patient, maxDepth, minSize int) Tree {
	if len(train) <= minSize || maxDepth == 0 {
		return Tree{Prediction: majorityClass(train)}
	}

	feature, threshold := bestSplit(train)
	if feature == -1 {
		return Tree{Prediction: majorityClass(train)}
	}

	left, right := splitDataByFeature(train, feature, threshold)
	if len(left) == 0 || len(right) == 0 {
		return Tree{Prediction: majorityClass(train)}
	}

	leftTree := trainTree(left, maxDepth-1, minSize)
	rightTree := trainTree(right, maxDepth-1, minSize)
	return Tree{
		Feature:    feature,
		Threshold:  threshold,
		Left:       &leftTree,
		Right:      &rightTree,
		Prediction: 0,
	}
}

func bestSplit(train []Patient) (int, float64) {
	bestFeature, bestThreshold := -1, 0.0
	bestGini := 1.0
	featureCount := len(train[0].Features)

	for f := 0; f < featureCount; f++ {
		if f == 4 || f == 7 {
			continue
		}
		values := make([]float64, len(train))
		for i, p := range train {
			values[i] = p.Features[f]
		}
		sort.Float64s(values)
		for i := 1; i < len(values); i++ {
			threshold := (values[i-1] + values[i]) / 2
			left, right := splitDataByFeature(train, f, threshold)
			gini := giniIndex(left, right)
			if gini < bestGini {
				bestGini, bestFeature, bestThreshold = gini, f, threshold
			}
		}
	}
	return bestFeature, bestThreshold
}

func splitDataByFeature(data []Patient, feature int, threshold float64) ([]Patient, []Patient) {
	left, right := []Patient{}, []Patient{}
	for _, p := range data {
		if p.Features[feature] <= threshold {
			left = append(left, p)
		} else {
			right = append(right, p)
		}
	}
	return left, right
}

func giniIndex(left, right []Patient) float64 {
	total := float64(len(left) + len(right))
	if total == 0 {
		return 0
	}
	pLeft := float64(len(left)) / total
	pRight := float64(len(right)) / total

	giniLeft := 1.0
	for _, class := range []int{0, 1, 2} {
		prop := float64(countClass(left, class)) / float64(len(left))
		if len(left) > 0 {
			giniLeft -= prop * prop
		}
	}

	giniRight := 1.0
	for _, class := range []int{0, 1, 2} {
		prop := float64(countClass(right, class)) / float64(len(right))
		if len(right) > 0 {
			giniRight -= prop * prop
		}
	}

	return pLeft*giniLeft + pRight*giniRight
}

func countClass(data []Patient, class int) int {
	count := 0
	for _, p := range data {
		if p.Label == class {
			count++
		}
	}
	return count
}

func majorityClass(data []Patient) int {
	counts := [3]int{}
	for _, p := range data {
		counts[p.Label]++
	}
	maxCount, maxClass := counts[0], 0
	for c := 1; c < 3; c++ {
		if counts[c] > maxCount {
			maxCount, maxClass = counts[c], c
		}
	}
	return maxClass
}

func trainRandomForest(train []Patient, nTrees, maxDepth, minSize int) RandomForest {
	forest := RandomForest{Trees: make([]Tree, nTrees)}
	for i := 0; i < nTrees; i++ {
		sample := make([]Patient, len(train))
		for j := range sample {
			sample[j] = train[rand.Intn(len(train))]
		}
		forest.Trees[i] = trainTree(sample, maxDepth, minSize)
	}
	return forest
}

func predictRandomForest(forest RandomForest, patient Patient) int {
	votes := [3]int{}
	for _, tree := range forest.Trees {
		pred := predictTree(tree, patient)
		votes[pred]++
	}
	maxVotes, maxClass := votes[0], 0
	for c := 1; c < 3; c++ {
		if votes[c] > maxVotes {
			maxVotes, maxClass = votes[c], c
		}
	}
	return maxClass
}

func predictTree(tree Tree, patient Patient) int {
	if tree.Left == nil && tree.Right == nil {
		return tree.Prediction
	}
	if patient.Features[tree.Feature] <= tree.Threshold {
		return predictTree(*tree.Left, patient)
	}
	return predictTree(*tree.Right, patient)
}

func evaluateModel(forest RandomForest, test []Patient) {
	confusion := [3][3]int{}
	for _, p := range test {
		pred := predictRandomForest(forest, p)
		confusion[p.Label][pred]++
	}

	fmt.Println("Confusion Matrix:")
	fmt.Println("    DM  DR  DN")
	for i := 0; i < 3; i++ {
		switch i {
		case 0:
			fmt.Print("DM: ")
		case 1:
			fmt.Print("DR: ")
		case 2:
			fmt.Print("DN: ")
		}
		for j := 0; j < 3; j++ {
			fmt.Printf("%2d  ", confusion[i][j])
		}
		fmt.Println()
	}

	correct := 0
	total := 0
	for i := 0; i < 3; i++ {
		truePos := float64(confusion[i][i])
		totalClass := 0
		for j := 0; j < 3; j++ {
			totalClass += confusion[i][j]
		}
		precisionDenom := 0
		for j := 0; j < 3; j++ {
			precisionDenom += confusion[j][i]
		}
		recall := 0.0
		if totalClass > 0 {
			recall = truePos / float64(totalClass)
		}
		precision := 0.0
		if precisionDenom > 0 {
			precision = truePos / float64(precisionDenom)
		}
		f1 := 0.0
		if precision+recall > 0 {
			f1 = 2 * precision * recall / (precision + recall)
		}
		accuracy := 0.0
		if totalClass > 0 {
			accuracy = truePos / float64(totalClass) * 100
		}
		fmt.Printf("Class %d (%s) - Accuracy: %.2f%%, Precision: %.2f, Recall: %.2f, F1-Score: %.2f\n",
			i, []string{"DM", "DR", "DN"}[i], accuracy, precision, recall, f1)
		correct += int(truePos)
		total += totalClass
	}
	overallAccuracy := float64(correct) / float64(total) * 100
	fmt.Printf("Overall Accuracy: %.2f%%\n", overallAccuracy)
}
