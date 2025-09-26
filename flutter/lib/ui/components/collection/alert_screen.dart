import 'package:flutter/material.dart';

class AlertsScreen extends StatelessWidget {
  const AlertsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final alerts = [
      {'card': 'Charizard', 'threshold': '\$80', 'type': 'Below'},
      {'card': 'Blastoise', 'threshold': '\$25', 'type': 'Below'},
    ];

    return Scaffold(
      appBar: AppBar(title: const Text('Price Alerts')),
      body: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(children: [
          ElevatedButton.icon(
            onPressed: () {
              ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Add alert UI not implemented in demo')));
            },
            icon: const Icon(Icons.add_alert),
            label: const Text('Add Alert'),
            style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF0F4C81)),
          ),
          const SizedBox(height: 12),
          Expanded(
            child: ListView.separated(
              itemCount: alerts.length,
              separatorBuilder: (_, __) => const Divider(color: Colors.white10),
              itemBuilder: (context, i) {
                final a = alerts[i];
                return ListTile(
                  leading: CircleAvatar(backgroundColor: Colors.white12, child: const Icon(Icons.card_membership)),
                  title: Text(a['card']!, style: const TextStyle(fontWeight: FontWeight.bold)),
                  subtitle: Text('${a['type']} ${a['threshold']}'),
                  trailing: const Icon(Icons.chevron_right),
                );
              },
            ),
          ),
        ]),
      ),
    );
  }
}
