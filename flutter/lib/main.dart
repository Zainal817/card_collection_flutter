import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'ui/components/services/api_service.dart';
import 'ui/components/landing.dart';
import 'ui/components/scan/scan.dart';
import 'ui/components/collection/collection.dart';
import 'ui/components/collection/card_detail.dart';

import 'ui/components/scan/bulk_scan_screen.dart';

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  final api = ApiService("http://10.0.2.2:8000"); // backend base URL

  MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "Card Collection",
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true),
      initialRoute: '/',
      routes: {
        '/': (context) => LandingPage(),
        '/scan': (context) => ScanOptionsPage(api: api, cameras: cameras),
        '/collection': (context) => CollectionScreen(),
        '/card': (context) => CardDetailScreen(cardData: {}),
      },
    );
  }
}
