import 'package:flutter/material.dart';

class PricingAlertsPage extends StatefulWidget {
  const PricingAlertsPage({super.key});

  @override
  State<PricingAlertsPage> createState() => _PricingAlertsPageState();
}

class _PricingAlertsPageState extends State<PricingAlertsPage> {
  final List<Map<String, dynamic>> _alerts = [
    {"name": "Gengar", "set": "Skyridge", "price": 670.0},
    {"name": "Blastoise", "set": "BASE Set", "price": 95.0},
  ];

  void _addAlert() {
    _showAlertDialog();
  }

  void _editAlert(int index) {
    _showAlertDialog(editIndex: index, existing: _alerts[index]);
  }

  void _showAlertDialog({int? editIndex, Map<String, dynamic>? existing}) {
    final formKey = GlobalKey<FormState>();
    final nameController = TextEditingController(text: existing?['name'] ?? "");
    final setController = TextEditingController(text: existing?['set'] ?? "");
    final priceController =
        TextEditingController(text: existing?['price']?.toString() ?? "");

    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: Text(editIndex == null ? "Add Alert" : "Edit Alert"),
        content: Form(
          key: formKey,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextFormField(
                controller: nameController,
                decoration: const InputDecoration(labelText: "Card Name"),
                validator: (value) =>
                    value == null || value.trim().isEmpty ? "Enter card name" : null,
              ),
              TextFormField(
                controller: setController,
                decoration: const InputDecoration(labelText: "Set"),
                validator: (value) =>
                    value == null || value.trim().isEmpty ? "Enter set name" : null,
              ),
              TextFormField(
                controller: priceController,
                decoration: const InputDecoration(labelText: "Price"),
                keyboardType: TextInputType.number,
                validator: (value) {
                  if (value == null || value.trim().isEmpty) {
                    return "Enter price";
                  }
                  final price = double.tryParse(value);
                  if (price == null || price <= 0) {
                    return "Enter valid price";
                  }
                  return null;
                },
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text("Cancel"),
          ),
          ElevatedButton(
            onPressed: () {
              if (formKey.currentState!.validate()) {
                final newAlert = {
                  "name": nameController.text.trim(),
                  "set": setController.text.trim(),
                  "price": double.parse(priceController.text.trim()),
                };

                setState(() {
                  if (editIndex == null) {
                    _alerts.add(newAlert);
                  } else {
                    _alerts[editIndex] = newAlert;
                  }
                });
                Navigator.pop(context);
              }
            },
            child: const Text("Save"),
          ),
        ],
      ),
    );
  }

  void _deleteAlert(int index) {
    setState(() {
      _alerts.removeAt(index);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1B263B),
      appBar: AppBar(
        title: const Text("Price Alerts"),
        backgroundColor: const Color(0xFF1B263B),
        actions: [
          TextButton(
            onPressed: _addAlert,
            child: const Text(
              "Add Alert",
              style: TextStyle(color: Colors.white),
            ),
          ),
        ],
      ),
      body: ListView.builder(
        itemCount: _alerts.length,
        itemBuilder: (context, index) {
          final alert = _alerts[index];
          return Dismissible(
            key: ValueKey(alert['name'] + alert['set']),
            direction: DismissDirection.endToStart,
            onDismissed: (_) => _deleteAlert(index),
            background: Container(
              color: Colors.red,
              alignment: Alignment.centerRight,
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: const Icon(Icons.delete, color: Colors.white),
            ),
            child: GestureDetector(
              onTap: () => _editAlert(index),
              child: Container(
                margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.blue.shade700,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(alert['name'],
                            style: const TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: Colors.white)),
                        Text(alert['set'],
                            style: const TextStyle(
                                fontSize: 14, color: Colors.white70)),
                      ],
                    ),
                    Text("\$${alert['price'].toStringAsFixed(2)}",
                        style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Colors.white)),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}
