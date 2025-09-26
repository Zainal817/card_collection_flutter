import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  final String baseUrl;
  ApiService(this.baseUrl);

  Future<Map<String, dynamic>> identifyCard(File file) async {
    final uri = Uri.parse("$baseUrl/identify");
    final request = http.MultipartRequest('POST', uri);
    request.files.add(await http.MultipartFile.fromPath('file', file.path));
    final streamed = await request.send();
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode != 200) throw Exception(resp.body);
    return json.decode(resp.body);  
  }

  Future<List<dynamic>> getCollections() async {
    final resp = await http.get(Uri.parse("$baseUrl/collections"));
    return json.decode(resp.body)["collections"];
  }

  Future<Map<String, dynamic>> getCard(String id) async {
    final resp = await http.get(Uri.parse("$baseUrl/card/$id"));
    return json.decode(resp.body);
  }

  Future<void> addToCollection(String cardId) async {
    await http.post(Uri.parse("$baseUrl/collections/add/$cardId"));
  }

  Future<List<dynamic>> getAlerts() async {
    final resp = await http.get(Uri.parse("$baseUrl/alerts"));
    return json.decode(resp.body)["alerts"];
  }
}
