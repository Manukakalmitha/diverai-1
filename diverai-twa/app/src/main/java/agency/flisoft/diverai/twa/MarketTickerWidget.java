package agency.flisoft.diverai.twa;

import android.app.PendingIntent;
import android.appwidget.AppWidgetManager;
import android.appwidget.AppWidgetProvider;
import android.content.Context;
import android.content.Intent;
import android.widget.RemoteViews;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import org.json.JSONObject;

public class MarketTickerWidget extends AppWidgetProvider {

    @Override
    public void onUpdate(Context context, AppWidgetManager appWidgetManager, int[] appWidgetIds) {
        for (int appWidgetId : appWidgetIds) {
            updateAppWidget(context, appWidgetManager, appWidgetId);
        }
    }

    static void updateAppWidget(Context context, AppWidgetManager appWidgetManager, int appWidgetId) {
        RemoteViews views = new RemoteViews(context.getPackageName(), R.layout.widget_market_ticker);

        // Click on symbol opens the app
        Intent intent = new Intent(context, LauncherActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(context, 0, intent, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        views.setOnClickPendingIntent(R.id.widget_ticker_symbol, pendingIntent);
        
        // Setup manual refresh on clicking price (Calls update again)
        Intent updateIntent = new Intent(context, MarketTickerWidget.class);
        updateIntent.setAction(AppWidgetManager.ACTION_APPWIDGET_UPDATE);
        updateIntent.putExtra(AppWidgetManager.EXTRA_APPWIDGET_IDS, new int[]{appWidgetId});
        PendingIntent updatePendingIntent = PendingIntent.getBroadcast(context, appWidgetId, updateIntent, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        views.setOnClickPendingIntent(R.id.widget_ticker_price, updatePendingIntent);

        // Fetch Data in background
        new Thread(() -> {
            String price = "---";
            String changeText = "";
            int changeColor = 0xFF888888; // Default gray

            try {
                // Fetch BTC Price with 24h Change
                URL url = new URL("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("GET");
                conn.setConnectTimeout(5000);
                conn.setReadTimeout(5000);
                conn.setRequestProperty("User-Agent", "DiverAI-Widget/1.0");

                if (conn.getResponseCode() == 200) {
                    BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                    StringBuilder result = new StringBuilder();
                    String line;
                    while ((line = reader.readLine()) != null) {
                        result.append(line);
                    }
                    reader.close();
                    
                    JSONObject json = new JSONObject(result.toString());
                    if (json.has("bitcoin")) {
                         JSONObject btcData = json.getJSONObject("bitcoin");
                         double p = btcData.getDouble("usd");
                         double c = btcData.optDouble("usd_24h_change", 0.0);

                         price = "$" + String.format("%,.0f", p);
                         
                         if (c > 0) {
                             changeText = "+" + String.format("%.2f", c) + "%";
                             changeColor = 0xFF4ADE80; // Green (Tailwind green-400 equivalent)
                         } else {
                             changeText = String.format("%.2f", c) + "%";
                             changeColor = 0xFFEF4444; // Red (Tailwind red-500 equivalent)
                         }
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }

            // Update the widget
            RemoteViews updateViews = new RemoteViews(context.getPackageName(), R.layout.widget_market_ticker);
            updateViews.setTextViewText(R.id.widget_ticker_price, price);
            updateViews.setTextViewText(R.id.widget_ticker_change, changeText);
            updateViews.setTextColor(R.id.widget_ticker_change, changeColor);

            // Re-bind click listeners
            updateViews.setOnClickPendingIntent(R.id.widget_ticker_symbol, pendingIntent);
            updateViews.setOnClickPendingIntent(R.id.widget_ticker_price, updatePendingIntent);
            
            appWidgetManager.updateAppWidget(appWidgetId, updateViews);
            
        }).start();
        
        // Set initial loading state
        views.setTextViewText(R.id.widget_ticker_price, "...");
        views.setTextViewText(R.id.widget_ticker_change, "");
        appWidgetManager.updateAppWidget(appWidgetId, views);
    }
}
