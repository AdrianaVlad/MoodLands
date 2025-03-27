using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace Assets.Scripts
{
    [System.Serializable]
    public class PredictionData
    {
        public string gender { get; set; }
        public int ageIndex { get; set; }
        public string age { get; set; }
        public string emotion { get; set; }
    }
}
