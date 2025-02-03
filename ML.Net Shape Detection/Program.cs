using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace ML.Net_Shape_Detection;

internal class Program
{
    static void Main(string[] args)
    {
        var builder = Host.CreateDefaultBuilder(args)
    .ConfigureAppConfiguration((context, config) =>
    {
        config.AddJsonFile("appsettings.json", optional: true, reloadOnChange: true);
    })
    .ConfigureServices((context, services) =>
    {
        // Register your services here
        services.AddTransient<Trainer>();
    });

        var host = builder.Build();

        using var serviceScope = host.Services.CreateScope();
        var services = serviceScope.ServiceProvider;

        var trainer = services.GetRequiredService<Trainer>();
        trainer.Go();
    }
}
