import 'package:flutter/material.dart';

class GradingEstimatorScreen extends StatelessWidget {
  const GradingEstimatorScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Grading Estimator'), backgroundColor: Colors.transparent, elevation: 0),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(children: [
          ClipRRect(borderRadius: BorderRadius.circular(12), child: Image.network('https://picsum.photos/600?random=15', height: 220, fit: BoxFit.cover)),
          const SizedBox(height: 12),
          const Text('Grading metrics', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
          const SizedBox(height: 12),
          _metricRow('Centering', '87/13'),
          _metricRow('Corners', '8/10'),
          _metricRow('Edges', '7/10'),
          _metricRow('Surface', '9/10'),
          const SizedBox(height: 12),
          const Text('Estimated Grade: PSA 9', style: TextStyle(fontWeight: FontWeight.bold, color: Colors.greenAccent)),
        ]),
      ),
    );
  }

  Widget _metricRow(String label, String value) {
    return Padding(padding: const EdgeInsets.symmetric(vertical: 8), child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [Text(label), Text(value, style: const TextStyle(fontWeight: FontWeight.bold))]));
  }
}
