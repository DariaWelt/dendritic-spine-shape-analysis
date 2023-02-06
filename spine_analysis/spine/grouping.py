import json
import random
from typing import Dict, Set, Iterable, List, Tuple

import numpy as np
from ipywidgets import widgets
from matplotlib import pyplot as plt

from spine_analysis.shape_metric.io_metric import SpineMetricDataset


class SpineGrouping:
    groups: Dict[str, Set[str]]
    samples: Set[str]
    outliers_label: str

    def __init__(self, samples: Iterable[str] = None, groups: Dict[str, Set[str]] = None,
                 outliers_label: str = None):
        if groups is None:
            groups = {}
        if samples is None:
            samples = set()
            for group in groups.values():
                samples = samples.union(group)
        self.samples = set(samples)
        self.groups = groups
        self.outliers_label = outliers_label

    @property
    def num_of_groups(self) -> int:
        return len(self.groups)

    @property
    def group_labels(self) -> Set[str]:
        return set(self.groups.keys())

    @property
    def sorted_group_labels(self) -> List[str]:
        labels = list(self.groups.keys())
        labels.sort()
        return labels

    @property
    def outlier_group(self) -> Set[str]:
        ng = self.samples
        for group in self.groups.values():
            ng = ng.difference(group)
        return ng

    @property
    def num_of_outlier(self) -> int:
        return len(self.samples) - len(self.outlier_group)

    @property
    def sample_size(self) -> int:
        return len(self.samples)

    @property
    def colors(self) -> Dict[str, Tuple[float, float, float, float]]:
        label_each = zip(self.groups.keys(),
                         np.linspace(0, 1, self.num_of_groups))
        return {group_label: plt.cm.Spectral(each)
                for (group_label, each) in label_each}

    @property
    def groups_with_outliers(self) -> Dict:
        output = self.groups.copy()
        output[self.outliers_label] = self.outlier_group
        return output

    @property
    def group_labels_with_outliers(self) -> Set[str]:
        output = self.group_labels
        output.add(self.outliers_label)
        return output

    @property
    def colors_with_outliers(self) -> Dict[str, Tuple[float, float, float, float]]:
        output = self.colors
        output[self.outliers_label] = (0.3, 0.3, 0.3, 1)
        return output

    def get_sorted_group(self, group_label: str) -> List[str]:
        spine_names = list(self.groups[group_label])
        spine_names.sort()
        return spine_names

    def get_spines_subset(self, spine_names: Iterable[str]) -> "SpineGrouping":
        groups = {label: set() for label in self.group_labels}
        groups[self.outliers_label] = set()
        for spine in spine_names:
            groups[self.get_group(spine)].add(spine)
        return SpineGrouping(spine_names, groups)

    def get_groups_subset(self, group_labels: Iterable[str]) -> "SpineGrouping":
        groups = {}
        for label in group_labels:
            groups[label] = self.groups[label]
        return SpineGrouping(groups=groups)

    def remove_samples(self, samples_to_removes: Set[str]):
        for sample in samples_to_removes:
            label = self.get_group(sample)
            if label is not None:
                self.groups[label].remove(sample)
        self.samples = self.samples.difference(samples_to_removes)

    @staticmethod
    def get_contested_samples(groupings: Iterable["SpineGrouping"], can_vote_outlier: bool = False) -> Set[str]:
        merged = SpineGrouping()

        # merged samples is union of samples from each grouping
        merged.samples = set().union(*[grouping.samples for grouping in groupings])

        # merged grouping contains all groups from each grouping
        for grouping in groupings:
            for label in grouping.groups.keys():
                merged.groups[label] = set()

        contested = set()

        # determine group for each sample
        if len(merged.groups) > 0:
            for spine_name in merged.samples:
                votes = {label: 0 for label in merged.group_labels}
                for grouping in groupings:
                    label = grouping.get_group(spine_name)
                    if label is not None or can_vote_outlier:
                        votes[label] += 1
                votes_sorted = [(label, vote_num) for (label, vote_num) in votes.items()]
                votes_sorted.sort(key=lambda label_vn: label_vn[1], reverse=True)
                if votes_sorted[0][1] == votes_sorted[1][1]:
                    contested.add(spine_name)

        return contested

    @staticmethod
    def accuracy(true_grouping: "SpineGrouping", predicted_grouping: "SpineGrouping") -> float:
        correct = 0
        for spine in true_grouping.samples:
            if true_grouping.get_group(spine) == predicted_grouping.get_group(spine):
                correct += 1
        return correct / true_grouping.sample_size

    @staticmethod
    def per_group_accuracy(true_grouping: "SpineGrouping", predicted_grouping: "SpineGrouping") -> Dict[str, float]:
        return {label: SpineGrouping.accuracy(true_grouping.get_groups_subset({label}),
                                              predicted_grouping.get_groups_subset({label}))
                for label in true_grouping.group_labels}

    @staticmethod
    def merge(groupings: Iterable["SpineGrouping"], can_vote_outlier: bool = False) -> "SpineGrouping":
        merged = SpineGrouping()

        # merged samples is union of samples from each grouping
        merged.samples = set().union(*[grouping.samples for grouping in groupings])

        # merged grouping contains all groups from each grouping
        for grouping in groupings:
            for label in grouping.groups.keys():
                merged.groups[label] = set()

        # determine group for each sample
        if len(merged.groups) > 0:
            for spine_name in merged.samples:
                votes = {label: 0 for label in merged.group_labels}
                for grouping in groupings:
                    label = grouping.get_group(spine_name)
                    if label is not None or can_vote_outlier:
                        votes[label] += 1
                votes_sorted = [(label, vote_num) for (label, vote_num) in votes.items()]
                votes_sorted.sort(key=lambda label_vn: label_vn[1], reverse=True)
                most_voted_label = votes_sorted[0][0]
                if most_voted_label is not None:
                    merged.groups[most_voted_label].add(spine_name)

        return merged

    def save(self, filename: str) -> None:
        with open(filename, "w") as file:
            json.dump({"groups": {label: list(group) for (label, group) in self.groups.items()},
                       "samples": list(self.samples), "outliers_label": self.outliers_label}, file)

    def load(self, filename: str) -> "SpineGrouping":
        with open(filename) as file:
            loaded = json.load(file)
            self.samples = set(s.replace('\\', '/').replace('0.025-0.025-0.1-dataset', '0.025 0.025 0.1 dataset') for s in loaded["samples"])
            self.groups = loaded["groups"]
            if "outliers_label" in loaded:
                self.outliers_label = loaded["outliers_label"]
            else:
                self.outliers_label = None
            for (key, group) in self.groups.items():
                self.groups[key] = set(g.replace('\\', '/').replace('0.025-0.025-0.1-dataset', '0.025 0.025 0.1 dataset') for g in group)
        return self

    def get_group(self, spine_name: str) -> str:
        for group_label, group in self.groups.items():
            if spine_name in group:
                return group_label
        return self.outliers_label

    def get_color(self, spine_name: str) -> Tuple[float, float, float, float]:
        group_label = self.get_group(spine_name)
        if group_label != self.outliers_label:
            return self.colors[group_label]
        return 0.0, 1, 1, 1

    def show(self, metrics: SpineMetricDataset,
             groups_to_show: Set[int] = None) -> widgets.Widget:
        out = widgets.Output()
        with out:
            self._show(metrics, groups_to_show)
            plt.show()

        return out

    def save_plot(self, metrics: SpineMetricDataset, filename: str) -> None:
        self._show(metrics)
        plt.savefig(filename)
        plt.clf()

    def _show(self, metrics: SpineMetricDataset, groups_to_show: Set[str] = None) -> None:
        def show_group(group_label: str, group: Set[str],
                       color: Tuple[float, float, float, float]) -> None:
            xy = reduced_data[[name_to_index[name] for name in group]]
            if xy.size > 0:
                plt.plot(
                    xy[:, 0],
                    xy[:, 1],
                    "o",
                    markerfacecolor=tuple(color),
                    markeredgecolor="k",
                    markersize=14,
                    label=f"{group_label}"
                )

        if groups_to_show is None:
            groups_to_show = set(self.groups.keys())

        colors = self.colors

        if metrics.as_array().shape[1] > 2:
            metrics = metrics.pca(2)

        reduced_data = metrics.as_array()
        name_to_index = {name: i for i, name in enumerate(metrics.ordered_spine_names)}

        for (group_label, group) in self.groups.items():
            color = colors[group_label] if group_label in groups_to_show else [
                0.69, 0.69, 0.69, 1]
            show_group(group_label, group, color)
        show_group(self.outliers_label, self.outlier_group, (0, 0, 0, 1))

        plt.title(f"Number of groups: {self.num_of_groups}")
        plt.legend(loc="upper right")
        plt.xlabel(metrics.metric_names[0])
        plt.ylabel(metrics.metric_names[1])

    def get_balanced_subset(self, size_ratio: float = 0.5) -> "SpineGrouping":
        new_groups = {}
        for (label, group) in self.groups.items():
            if len(group) > 0:
                list_group = list(group)
                list_group.sort()
                random.shuffle(list_group)
                new_groups[label] = set(list_group[:int(len(group) * size_ratio) + 1])
            else:
                new_groups[label] = set()

        new_samples = set()
        for group in new_groups.values():
            new_samples = new_samples.union(group)
        outliers = self.outlier_group
        new_samples = new_samples.union(list(outliers)[:int(len(outliers) * size_ratio) + 1])

        return SpineGrouping(new_samples, new_groups)

    def intersection_ratios(self, other: "SpineGrouping", normalize: bool = True) -> Dict[str, Dict[str, float]]:
        intersections = {}
        for i, (self_label, self_group) in enumerate(self.groups_with_outliers.items()):
            if len(self_group) == 0:
                continue
            intersections[self_label] = {}
            for j, (other_label, other_group) in enumerate(other.groups_with_outliers.items()):
                if len(other_group) == 0:
                    value = 0
                else:
                    value = len(self_group.intersection(other_group)) / len(self_group)
                    if normalize:
                        value /= len(other_group)
                intersections[self_label][other_label] = value
            if normalize:
                intersection_sum = sum(value for value in intersections[self_label].values())
                for other_label in intersections[self_label].keys():
                    intersections[self_label][other_label] /= intersection_sum

        return intersections
